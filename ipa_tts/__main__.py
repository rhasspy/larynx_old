#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import logging
import sys
import typing
import wave
from dataclasses import dataclass
from pathlib import Path

import gruut

_LOGGER = logging.getLogger("ipa_tts")

_DIR = Path(__file__).parent

# -----------------------------------------------------------------------------

_TEST_SENTENCES = {
    "nl": [
        "Hoe laat is het?",
        "Nog een prettige dag toegewenst.",
        "Kunt u wat langzamer praten, alstublieft?",
        "Van Harte Gefeliciteerd met je verjaardag!",
        "Moeder sneed zeven scheve sneden brood.",
    ]
}

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    args.func(args)


# -----------------------------------------------------------------------------


@dataclass
class DatasetItem:
    """Single audio item from a dataset"""

    id: str
    text: str
    wav_path: Path


@dataclass
class AudioStats:
    mel_sum: float = 0
    mel_square_sum: float = 0
    linear_sum: float = 0
    linear_square_sum: float = 0
    N: int = 0


def _compute_phonemes(
    dataset_items: typing.Dict[str, DatasetItem],
    gruut_lang: gruut.Language,
    phonemes: typing.Dict[str, int],
    model_dir: Path,
    phoneme_cache_dir: Path,
):
    """Tokenize and phonemize transcripts"""
    import numpy as np
    import phonetisaurus

    _LOGGER.debug("Generating phonemes")

    # Tokenize/clean transcripts
    def tokenize(item: DatasetItem) -> typing.List[str]:
        clean_words = []
        for sentence in gruut_lang.tokenizer.tokenize(item.text):
            clean_words.extend(sentence.clean_words)

        return (item.id, clean_words)

    _LOGGER.debug("Tokenizing...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        id_clean_words = list(executor.map(tokenize, dataset_items.values()))

    # Load missing words
    lexicon = gruut_lang.phonemizer.lexicon

    missing_words_path = model_dir / "missing_words.txt"
    if missing_words_path.is_file():
        _LOGGER.debug("Loading missing words from %s", missing_words_path)
        with open(missing_words_path, "r") as missing_words_file:
            gruut.utils.load_lexicon(missing_words_file, lexicon=lexicon)

    # Guess missing words
    missing_words: typing.Set[str] = set()

    for item_id, item_clean_words in id_clean_words:
        for word in item_clean_words:
            if (word not in lexicon) and gruut_lang.tokenizer.is_word(word):
                missing_words.add(word)

    if missing_words:
        _LOGGER.debug("Guessing pronunciations for %s word(s)", len(missing_words))
        word_prons = phonetisaurus.predict(
            missing_words, gruut_lang.phonemizer.g2p_model_path, nbest=1
        )

        guessed_words_path = model_dir / "guessed_words.txt"
        with open(guessed_words_path, "w") as guessed_words_file:
            for word, pron in word_prons:
                # Assuming only one pronunciation
                lexicon[word] = [pron]
                print(word, " ".join(pron), file=guessed_words_file)

        _LOGGER.debug(
            "Wrote guessed words to %s. Move to %s if they're correct.",
            guessed_words_path,
            missing_words_path,
        )

    # Phonemize clean words
    def phonemize(item_clean_words: typing.Tuple[str, typing.List[str]]) -> np.ndarray:
        item_id, clean_words = item_clean_words
        sequence = []

        # Choose first pronunciation for each word
        word_phonemes = [
            wp[0]
            for wp in gruut_lang.phonemizer.phonemize(
                clean_words, word_indexes=True, word_breaks=True
            )
            if wp
        ]

        # Convert to integer sequence.
        # Drop unknown phonemes.
        sequence.extend(
            phonemes[p] for ps in word_phonemes for p in ps if p in phonemes
        )

        return item_id, np.array(sequence, dtype=np.int32)

    _LOGGER.debug("Phonemizing...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        id_phonemes = executor.map(phonemize, id_clean_words)

    # Save phonemes to cache as numpy arrays
    num_saved = 0
    for item_id, item_phonemes in id_phonemes:
        item_phonemes_path = (phoneme_cache_dir / f"{item_id}_phoneme").with_suffix(
            ".npy"
        )
        with open(item_phonemes_path, "wb") as item_phonemes_file:
            np.save(item_phonemes_file, item_phonemes)

        num_saved += 1

    _LOGGER.debug("Finished writing phonemes for %s item(s)", num_saved)


def _compute_audio_stats(
    dataset_items: typing.Dict[str, DatasetItem],
    tts_config: typing.Dict[str, typing.Any],
    tts_stats_path: Path,
):
    """Compute audio statistics in parallel"""
    import numpy as np

    from TTS.utils.audio import AudioProcessor

    # Prevent attempt to load non-existent stats
    tts_config["audio"]["stats_path"] = None

    tts_ap = AudioProcessor(**tts_config["audio"])

    def get_stats(item: DatasetItem) -> AudioStats:
        """Compute audio statistics of a WAV"""
        wav = tts_ap.load_wav(item.wav_path)
        linear = tts_ap.spectrogram(wav)
        mel = tts_ap.melspectrogram(wav)

        return AudioStats(
            N=mel.shape[1],
            mel_sum=mel.sum(1),
            linear_sum=linear.sum(1),
            mel_square_sum=(mel ** 2).sum(axis=1),
            linear_square_sum=(linear ** 2).sum(axis=1),
        )

    # Compute in parallel and then aggregate
    _LOGGER.debug("Computing audio stats...")
    sum_stats = AudioStats()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for item_stats in executor.map(get_stats, dataset_items.values()):
            sum_stats.N += item_stats.N
            sum_stats.mel_sum += item_stats.mel_sum
            sum_stats.linear_sum += item_stats.linear_sum
            sum_stats.mel_square_sum += item_stats.mel_square_sum
            sum_stats.linear_square_sum += item_stats.linear_square_sum

    # Average aggregate stats
    mel_mean = sum_stats.mel_sum / sum_stats.N
    linear_mean = sum_stats.linear_sum / sum_stats.N

    stats = {
        "mel_mean": mel_mean,
        "mel_std": np.sqrt(sum_stats.mel_square_sum / sum_stats.N - mel_mean ** 2),
        "linear_mean": linear_mean,
        "linear_std": np.sqrt(
            sum_stats.linear_square_sum / sum_stats.N - linear_mean ** 2
        ),
    }

    _LOGGER.debug("Audio stats: %s", stats)

    stats["audio_config"] = tts_config["audio"]
    np.save(tts_stats_path, stats, allow_pickle=True)

    _LOGGER.debug("Wrote audio stats to %s", tts_stats_path)


def do_init(args):
    """Initialize a model directory for training"""
    import json5
    from gruut_ipa import IPA

    dataset_items: typing.Dict[str, DatasetItem] = {}

    model_dir = Path(args.model)
    language = args.language
    dataset_dir = Path(args.dataset)
    model_name = args.name or model_dir.name

    _LOGGER.debug("Loading gruut language %s", language)
    gruut_lang = gruut.Language.load(language)
    assert gruut_lang, f"Unsupported language: {gruut_lang}"

    # Create base output directory
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = dataset_dir / "metadata.csv"
    _LOGGER.debug("Loading metadata file from %s", metadata_path)
    _LOGGER.debug("Expecting WAV files in %s", dataset_dir)

    with open(metadata_path, "r") as metadata_file:
        for line in metadata_file:
            line = line.strip()
            if line:
                item_id, item_text = line.split("|", maxsplit=1)
                dataset_items[item_id] = DatasetItem(
                    id=item_id,
                    text=item_text,
                    wav_path=(dataset_dir / item_id).with_suffix(".wav"),
                )

    assert dataset_items, "No items in dataset"
    _LOGGER.debug("Loaded transcripts for %s item(s)", len(dataset_items))

    # -------------
    # Phoneme Cache
    # -------------

    pad = "_"

    # Always include pad and break symbols.
    # In the future, intontation should also be added.
    phonemes_list = [
        pad,
        IPA.BREAK_MINOR.value,
        IPA.BREAK_MAJOR.value,
        IPA.BREAK_WORD.value,
    ] + sorted([p.text for p in gruut_lang.phonemes])

    # Index where actual model phonemes start
    phoneme_offset = 1

    # Map to indexes
    phonemes = {p: i for i, p in enumerate(phonemes_list)}

    _LOGGER.debug("Phonemes: %s", phonemes)

    phoneme_cache_dir = model_dir / "phoneme_cache"
    phoneme_cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_phonemes:
        _compute_phonemes(
            dataset_items, gruut_lang, phonemes, model_dir, phoneme_cache_dir
        )

    # ----------
    # TTS Config
    # ----------

    # Get sample rate from first WAV file
    first_item = next(iter(dataset_items.values()))
    with open(first_item.wav_path, "rb") as first_wav_file:
        with wave.open(first_wav_file, "rb") as first_wav:
            sample_rate = first_wav.getframerate()

    _LOGGER.debug("Assuming sample rate is %s Hz", sample_rate)

    # Path to MozillaTTS submodule
    tts_dir = _DIR.parent / "TTS"

    # Load TTS model base config
    tts_configs_dir = tts_dir / "TTS" / "tts" / "configs"

    model_type = args.model_type.strip().lower()
    if model_type == "tacotron2":
        tts_config_in_path = tts_configs_dir / "config.json"
    elif model_type == "glowtts":
        tts_config_in_path = tts_configs_dir / "glow_tts_gated_conv.json"
    else:
        raise ValueError(f"Unexpected model type: {model_type}")

    _LOGGER.debug("Loading TTS config template from %s", tts_config_in_path)
    with open(tts_config_in_path, "r") as tts_config_file:
        tts_config = json5.load(tts_config_file)

    # Patch configuration and write to output directory
    tts_config["run_name"] = model_name

    tts_config["audio"]["sample_rate"] = sample_rate
    tts_config["audio"]["do_trim_silence"] = False
    tts_config["audio"]["signal_norm"] = True

    tts_config["output_path"] = str(model_dir / "model")
    tts_config["phoneme_cache_path"] = str(phoneme_cache_dir)
    tts_config["phoneme_language"] = language
    tts_config["phoneme_backend"] = "gruut"

    # Align faster
    tts_config["use_forward_attn"] = True

    # Disable DDC
    tts_config["double_decoder_consistency"] = False

    # Disable global style tokens
    tts_config["use_gst"] = False
    tts_config["gst"]["gst_use_speaker_embedding"] = False

    # Disable speaker embedding
    tts_config["use_external_speaker_embedding_file"] = False
    tts_config["external_speaker_embedding_file"] = None
    tts_config["use_speaker_embedding"] = False

    # Use custom phonemes
    tts_config["use_phonemes"] = True
    tts_config["enable_eos_bos_chars"] = False
    tts_config["characters"] = {
        "pad": pad,
        "eos": "~",
        "bos": "^",
        "phonemes": phonemes_list[phoneme_offset:],
        "characters": "",
        "punctuations": "",
        "eos_bos_phonemes": False,
        "sort_phonemes": False,
    }

    tts_config["datasets"] = [
        {
            "name": "ipa_tts",
            "path": str(dataset_dir),
            "meta_file_train": "metadata.csv",
            "meta_file_val": None,
        }
    ]

    # Test sentences
    test_sentences = _TEST_SENTENCES.get(language)
    if test_sentences:
        test_sentences_path = model_dir / "test_sentences.txt"
        with open(test_sentences_path, "w") as test_sentences_file:
            for sentence in test_sentences:
                print(sentence, file=test_sentences_file)

        tts_config["test_sentences_file"] = str(test_sentences_path)

    # -------------------
    # Compute Audio Stats
    # -------------------

    tts_stats_path = str(model_dir / "scale_stats.npy")

    if not args.skip_audio_stats:
        _compute_audio_stats(dataset_items, tts_config, tts_stats_path)

    tts_config["audio"]["stats_path"] = str(tts_stats_path)

    # Write TTS config
    tts_config_out_path = model_dir / "config.json"
    with open(tts_config_out_path, "w") as tts_config_file:
        json.dump(tts_config, tts_config_file, indent=4, ensure_ascii=False)

    _LOGGER.debug("Wrote TTS config to %s", tts_config_out_path)

    # --------------
    # Vocoder config
    # --------------

    vocoder_dir = model_dir / "vocoder"
    vocoder_dir.mkdir(parents=True, exist_ok=True)

    vocoder_config_in_path = (
        tts_dir / "TTS" / "vocoder" / "configs" / "multiband_melgan_config.json"
    )

    _LOGGER.debug("Loading vocoder config template from %s", vocoder_config_in_path)
    with open(vocoder_config_in_path, "r") as vocoder_config_file:
        vocoder_config = json5.load(vocoder_config_file)

    # Check sample rate
    vocoder_sample_rate = vocoder_config["audio"]["sample_rate"]
    if sample_rate == vocoder_sample_rate:
        # Use data as is
        _LOGGER.debug("Using existing data at %s", dataset_dir)
        vocoder_config["data_path"] = str(dataset_dir)
    else:
        # TODO: Need to resample
        assert False, "Need to resample"
        _LOGGER.warning(
            "Vocoder sample rate is %s while TTS sample rate is %s",
            vocoder_sample_rate,
            sample_rate,
        )

        data_dir = vocoder_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

    # Patch vocoder config
    vocoder_config["run_name"] = model_name
    vocoder_config["output_path"] = str(vocoder_dir / "model")

    # Use same audio configuration as voice
    vocoder_config["audio"] = tts_config["audio"]

    if args.vocoder_batch_size:
        vocoder_config["batch_size"] = args.vocoder_batch_size

    vocoder_config_out_path = vocoder_dir / "config.json"
    with open(vocoder_config_out_path, "w") as vocoder_out_file:
        json.dump(vocoder_config, vocoder_out_file, indent=4, ensure_ascii=False)

    _LOGGER.debug("Wrote vocoder config to %s", vocoder_config_out_path)


# -----------------------------------------------------------------------------


def do_synthesize(args):
    """Synthesize WAV data from text"""
    pass

    # from .synthesize import Synthesizer

    # synthesizer = Synthesizer(
    #     tts_model_path=args.model,
    #     tts_config_path=args.config,
    #     vocoder_model_path=args.vocoder_model,
    #     vocoder_config_path=args.vocoder_config,
    #     phonemes_path=args.phonemes,
    #     use_cuda=args.use_cuda,
    # )

    # try:
    #     for text in sys.stdin:
    #         text = text.strip()
    #         if text:
    #             wav_bytes = synthesizer.synthesize(text)
    #             subprocess.run(["play", "-q", "-t" "wav", "-"], input=wav_bytes)
    # except KeyboardInterrupt:
    #     pass


# -----------------------------------------------------------------------------


def do_train(args):
    """Train a new text to speech voice"""
    pass

    # from .config import Config
    # from .dataset import Dataset, DatasetItem
    # from .train import train_dataset

    # config = Config(name="kathy", language="en-us")

    # items = []
    # metadata_path = Path(args.metadata)
    # wav_dir = metadata_path.parent / "wav"

    # _LOGGER.debug("Loading dataset from %s", metadata_path)
    # with open(metadata_path, "r") as metadata_file:
    #     for line in metadata_file:
    #         line = line.strip()
    #         if not line:
    #             continue

    #         item_obj = json.loads(line)
    #         phoneme_indexes = []
    #         for word_pron in item_obj["pronunciation"]:
    #             phoneme_indexes.extend(
    #                 [config.phoneme_to_id[phoneme] for phoneme in word_pron if phoneme]
    #             )

    #         # Get WAV path
    #         item_id = item_obj["id"]
    #         wav_path = (wav_dir / item_id).with_suffix(".wav")

    #         items.append(
    #             DatasetItem(phoneme_indexes=phoneme_indexes, wav_path=wav_path)
    #         )

    # dataset = Dataset(config, items)
    # train_dataset(dataset)


# -----------------------------------------------------------------------------


def do_serve(args):
    """Run web server for synthesis"""
    pass

    # from .server import get_app
    # from .synthesize import Synthesizer

    # synthesizer = Synthesizer(
    #     tts_model_path=args.model,
    #     tts_config_path=args.config,
    #     vocoder_model_path=args.vocoder_model,
    #     vocoder_config_path=args.vocoder_config,
    #     phonemes_path=args.phonemes,
    #     use_cuda=args.use_cuda,
    # )

    # app = get_app(synthesizer)
    # app.run(host=args.host, port=args.port)


# -----------------------------------------------------------------------------


def do_phonemize(args):
    from TTS.utils.io import load_config
    from TTS.tts.utils.text import make_symbols, phoneme_to_sequence

    c = load_config(args.config)
    _, phonemes = make_symbols(**c.characters)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        line_indexes = phoneme_to_sequence(
            line,
            [c.text_cleaner],
            language=c.phoneme_language,
            enable_eos_bos=False,
            tp=c.characters if "characters" in c.keys() else None,
            backend=c.phoneme_backend,
        )

        line_phonemes = [phonemes[i] for i in line_indexes]

        print(line_phonemes)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="ipa-tts")

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # ----
    # init
    # ----
    init_parser = sub_parsers.add_parser(
        "init", help="Initialize a model directory for a dataset"
    )
    init_parser.add_argument("model", help="Path to model base directory")
    init_parser.add_argument(
        "--language", required=True, help="Language for model (e.g. en-us)"
    )
    init_parser.add_argument(
        "--dataset", required=True, help="Path to dataset directory"
    )
    init_parser.add_argument(
        "--name", help="Name of model (default: model directory name)"
    )
    init_parser.add_argument(
        "--model-type",
        default="tacotron2",
        choices=["tacotron2", "glowtts"],
        help="Type of MozillaTTS model (default: tacotron2)",
    )
    init_parser.add_argument(
        "--skip-phonemes", action="store_true", help="Skip phoneme computation"
    )
    init_parser.add_argument(
        "--skip-audio-stats",
        action="store_true",
        help="Skip audio statistics computation",
    )
    init_parser.add_argument(
        "--vocoder-batch-size",
        type=int,
        help="Batch size for vocoder (default: config value)",
    )
    init_parser.set_defaults(func=do_init)

    # ---------
    # phonemize
    # ---------
    phonemize_parser = sub_parsers.add_parser(
        "phonemize", help="Path to TTS JSON configuration file"
    )
    phonemize_parser.add_argument(
        "--config", required=True, help="Path to TTS JSON configuration file"
    )
    phonemize_parser.set_defaults(func=do_phonemize)

    # ----------
    # synthesize
    # ----------
    synthesize_parser = sub_parsers.add_parser(
        "synthesize", help="Generate WAV data for IPA phonemes"
    )
    synthesize_parser.add_argument("text", nargs="*", help="Sentences to synthesize")
    synthesize_parser.add_argument(
        "--model", required=True, help="Path to TTS model checkpoint"
    )
    synthesize_parser.add_argument(
        "--config", help="Path to TTS model JSON config file"
    )
    synthesize_parser.add_argument(
        "--vocoder-model", help="Path to vocoder model checkpoint"
    )
    synthesize_parser.add_argument(
        "--vocoder-config", help="Path to vocoder model JSON config file"
    )
    synthesize_parser.add_argument("--phonemes", help="Path to phonemes text file")
    synthesize_parser.add_argument(
        "--use-cuda", action="store_true", help="Use GPU (CUDA) for synthesis"
    )
    synthesize_parser.set_defaults(func=do_synthesize)

    # -----
    # train
    # -----
    train_parser = sub_parsers.add_parser(
        "train", help="Train a new model or tune an existing model"
    )
    train_parser.set_defaults(func=do_train)
    train_parser.add_argument("metadata", help="JSONL file with training metadata")

    # -----
    # serve
    # -----
    serve_parser = sub_parsers.add_parser("serve", help="Run web server for synthesis")
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host for web server (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=5002, help="Port for web server (default: 5002)"
    )
    serve_parser.add_argument(
        "--model", required=True, help="Path to TTS model checkpoint"
    )
    serve_parser.add_argument("--config", help="Path to TTS model JSON config file")
    serve_parser.add_argument(
        "--vocoder-model", help="Path to vocoder model checkpoint"
    )
    serve_parser.add_argument(
        "--vocoder-config", help="Path to vocoder model JSON config file"
    )
    serve_parser.add_argument("--phonemes", help="Path to phonemes text file")
    serve_parser.add_argument(
        "--use-cuda", action="store_true", help="Use GPU (CUDA) for synthesis"
    )
    serve_parser.set_defaults(func=do_serve)

    # Shared arguments
    for sub_parser in [
        init_parser,
        synthesize_parser,
        train_parser,
        serve_parser,
        phonemize_parser,
    ]:
        sub_parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to console"
        )

    return parser.parse_args()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
