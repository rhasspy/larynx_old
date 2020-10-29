#!/usr/bin/env python3
"""Command-line interface to larynx"""
import argparse
import concurrent.futures
import csv
import json
import logging
import os
import re
import string
import subprocess
import sys
import typing
import wave
from dataclasses import dataclass
from pathlib import Path

import gruut

_LOGGER = logging.getLogger("larynx")

_DIR = Path(__file__).parent

# -----------------------------------------------------------------------------

_TEST_SENTENCES = {
    "nl": [
        "Hoe laat is het?",
        "Nog een prettige dag toegewenst.",
        "Kunt u wat langzamer praten, alstublieft?",
        "Van Harte Gefeliciteerd met je verjaardag!",
        "Moeder sneed zeven scheve sneden brood.",
    ],
    "de-de": [
        "Können Sie bitte langsamer sprechen?",
        "Mir geht es gut, danke!",
        "Haben Sie ein vegetarisches Gericht?",
        "Ich bin allergisch.",
        "Fischers Fritze fischt frische Fische; Frische Fische fischt Fischers Fritze.",
    ],
    "fr-fr": [
        "Pourriez-vous parler un peu moins vite?",
        "Je suis allergique.",
        "Est-ce que vous pourriez l'écrire?",
        "Avez-vous des plats végétariens?",
        "Si mon tonton tond ton tonton, ton tonton sera tondu.",
    ],
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
    """Audio statistics for scale_stats.npy"""

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
        try:
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
        except Exception as e:
            _LOGGER.exception(str(item))
            raise e

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
    import numpy as np
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
                    id=item_id, text=item_text, wav_path=dataset_dir / f"{item_id}.wav"
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

    # Write phonemes to a text file
    phonemes_text_path = model_dir / "phonemes.txt"
    with open(phonemes_text_path, "w") as phonemes_text_file:
        for phoneme_idx, phoneme in enumerate(phonemes_list):
            print(phoneme_idx, phoneme, file=phonemes_text_file)

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

    # Write phonemized sentences
    if phoneme_cache_dir.is_dir():
        dataset_phonemes_path = model_dir / "dataset_phonemes.csv"

        with open(dataset_phonemes_path, "w") as dataset_phonemes_file:
            phonemes_writer = csv.writer(dataset_phonemes_file, delimiter="|")
            phonemes_writer.writerow(("id", "text", "phonemes"))

            for phoneme_path in phoneme_cache_dir.glob("*.npy"):
                item_id = re.sub("_phoneme$", "", phoneme_path.stem)
                sequence = np.load(phoneme_path, allow_pickle=True)
                actual_phonemes = [phonemes_list[index] for index in sequence]

                item = dataset_items.get(item_id)
                if item:
                    actual_phonemes_str = " ".join(actual_phonemes)
                    phonemes_writer.writerow((item_id, item.text, actual_phonemes_str))
                else:
                    _LOGGER.warning(
                        "Item %s is in phoneme cache but not in dataset", item_id
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

    if "gst" not in tts_config:
        tts_config["gst"] = {}

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

    # Patch vocoder config
    vocoder_config["data_path"] = str(dataset_dir)
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


def do_compute_stats(args):
    """Compute audio statistics for dataset(s)"""
    import json5

    model_dir = Path(args.model)
    dataset_dir = Path(args.dataset)

    tts_config_path = model_dir / "config.json"
    with open(tts_config_path, "r") as tts_config_file:
        tts_config = json5.load(tts_config_file)

    # Load dataset
    dataset_items: typing.Dict[str, DatasetItem] = {}

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
                    id=item_id, text=item_text, wav_path=dataset_dir / f"{item_id}.wav"
                )

    # Compute stats
    tts_stats_path = str(model_dir / "scale_stats.npy")
    _compute_audio_stats(dataset_items, tts_config, tts_stats_path)


# -----------------------------------------------------------------------------


def do_synthesize(args):
    """Synthesize WAV data from text"""
    from .synthesize import Synthesizer

    # Guess missing config paths
    if not args.config:
        args.config = os.path.join(os.path.dirname(args.model), "config.json")

    if args.vocoder_model and not args.vocoder_config:
        args.vocoder_config = os.path.join(
            os.path.dirname(args.vocoder_model), "config.json"
        )

    # Convert to paths
    if args.output_file:
        args.output_file = Path(args.output_file)

    if args.output_dir:
        args.output_dir = Path(args.output_dir)

    # Load synthesizer
    synthesizer = Synthesizer(
        config_path=args.config,
        model_path=args.model,
        use_cuda=args.use_cuda,
        vocoder_path=args.vocoder_model,
        vocoder_config_path=args.vocoder_config,
    )

    synthesizer.load()

    # Fix logging (something in MozillaTTS is changing the level)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Accents
    accent_lang = None
    phoneme_map: typing.Dict[str, typing.List[str]] = {}
    if args.accent_language:
        source_language = synthesizer.config["phoneme_language"]
        accent_lang = gruut.Language.load(args.accent_language)
        phoneme_map = accent_lang.accents[source_language]

    # Args or stdin
    if args.text:
        texts = args.text
    else:
        texts = sys.stdin

    try:
        # Process sentences line by line
        for text in texts:
            text = text.strip()
            if not text:
                continue

            original_text = text
            text_is_phonemes = args.phonemes

            if text_is_phonemes:
                # Interpret text input as phonemes with a separator
                text = text.split(args.phoneme_separator)
            elif accent_lang and phoneme_map:
                # Interpret text in the accent language, map to phonemes in
                # the voice language.
                text_phonemes = []
                for sentence in accent_lang.tokenizer.tokenize(text):
                    # Choose first pronunciation for each word
                    word_phonemes = [
                        wp[0]
                        for wp in accent_lang.phonemizer.phonemize(
                            sentence.clean_words, word_indexes=True, word_breaks=True
                        )
                        if wp
                    ]

                    # Do phoneme mapping
                    for wp in word_phonemes:
                        for p in wp:
                            p2 = phoneme_map.get(p)
                            if p2:
                                text_phonemes.extend(p2)
                            else:
                                text_phonemes.append(p)

                _LOGGER.debug(text_phonemes)
                text = text_phonemes
                text_is_phonemes = True

            # -------------------------------------------------------------

            # Do synthesis
            wav_bytes = synthesizer.synthesize(text, text_is_phonemes=text_is_phonemes)

            if args.output_file:
                # Write to single file.
                # Will overwrite if multiple sentences.
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                args.output_file.write_bytes(wav_bytes)
                _LOGGER.debug("Wrote %s", args.output_file)
            elif args.output_dir:
                # Write to directory.
                # Name WAV file after text input.
                file_name = original_text.replace(" ", "_")
                file_name = (
                    file_name.translate(
                        str.maketrans("", "", string.punctuation.replace("_", ""))
                    )
                    + ".wav"
                )

                args.output_dir.mkdir(parents=True, exist_ok=True)
                file_path = Path(args.output_dir / file_name)
                file_path.write_bytes(wav_bytes)
                _LOGGER.debug("Wrote %s", file_path)
            else:
                # Play using sox
                subprocess.run(
                    ["play", "-q", "-t", "wav", "-"], input=wav_bytes, check=True
                )
    except KeyboardInterrupt:
        # CTRL + C
        pass


# -----------------------------------------------------------------------------


def do_serve(args):
    """Run web server for synthesis"""
    from larynx.server import get_app
    from larynx.synthesize import Synthesizer

    # Guess missing config paths
    if not args.config:
        args.config = os.path.join(os.path.dirname(args.model), "config.json")

    if args.vocoder_model and not args.vocoder_config:
        args.vocoder_config = os.path.join(
            os.path.dirname(args.vocoder_model), "config.json"
        )

    # Load synthesizer
    synthesizer = Synthesizer(
        config_path=args.config,
        model_path=args.model,
        use_cuda=args.use_cuda,
        vocoder_path=args.vocoder_model,
        vocoder_config_path=args.vocoder_config,
    )

    synthesizer.load()

    # Fix logging (something in MozillaTTS is changing the level)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Load language
    gruut_lang = gruut.Language.load(synthesizer.config.phoneme_language)

    # Run web server
    app = get_app(synthesizer, gruut_lang=gruut_lang, cache_dir=args.cache_dir)
    app.run(host=args.host, port=args.port)


# -----------------------------------------------------------------------------


def do_phonemize(args):
    """Generate phonemes for text using config"""
    from TTS.utils.io import load_config
    from TTS.tts.utils.text import make_symbols, phoneme_to_sequence

    c = load_config(args.config)
    _, phonemes = make_symbols(**c.characters)

    if args.text:
        # Use arguments
        texts = args.text
    else:
        # Use stdin
        texts = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    for line in texts:
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

        print(args.separator.join(line_phonemes))


# -----------------------------------------------------------------------------


def do_verify_phonemes(args):
    """Verify that phoneme cache matches what gruut would produce"""
    import numpy as np
    from TTS.utils.io import load_config
    from TTS.tts.utils.text import make_symbols

    _LOGGER.debug("Loading gruut language %s", args.language)
    gruut_lang = gruut.Language.load(args.language)
    assert gruut_lang, f"Unsupported language: {gruut_lang}"

    # Load config
    c = load_config(args.config)
    output_path = Path(c.output_path)
    phoneme_cache_dir = Path(c.phoneme_cache_path)
    _, phonemes = make_symbols(**c.characters)

    # Offset for pad
    phoneme_to_id = {p: (i + 1) for i, p in enumerate(phonemes)}

    # Add pad
    phoneme_to_id["_"] = 0

    # Load lexicon and missing words
    lexicon = gruut_lang.phonemizer.lexicon

    missing_words_path = output_path / "missing_words.txt"
    if missing_words_path.is_file():
        _LOGGER.debug("Loading missing words from %s", missing_words_path)
        with open(missing_words_path, "r") as missing_words_file:
            gruut.utils.load_lexicon(missing_words_file, lexicon=lexicon)

    # Load metadata
    id_to_text = {}
    for ds in c.datasets:
        metadata_path = Path(ds["path"]) / ds["meta_file_train"]
        with open(metadata_path, "r") as metadata_file:
            for line in metadata_file:
                line = line.strip()
                if line:
                    item_id, item_text = line.split("|", maxsplit=1)
                    id_to_text[item_id] = item_text

    id_to_phonemes = {}
    for phoneme_path in phoneme_cache_dir.glob("*.npy"):
        item_id = re.sub("_phoneme$", "", phoneme_path.stem)
        _LOGGER.debug("Processing %s (id=%s)", phoneme_path, item_id)

        sequence = np.load(phoneme_path, allow_pickle=True)
        actual_phonemes = [phonemes[index] for index in sequence]

        expected_phonemes = id_to_phonemes.get(item_id)
        if not expected_phonemes:
            # Compute expected phonmemes
            expected_phonemes = []

            item_text = id_to_text[item_id]
            for sentence in gruut_lang.tokenizer.tokenize(item_text):
                # Choose first pronunciation for each word
                word_phonemes = [
                    wp[0]
                    for wp in gruut_lang.phonemizer.phonemize(
                        sentence.clean_words, word_indexes=True, word_breaks=True
                    )
                    if wp
                ]

            expected_phonemes.extend(p for ps in word_phonemes for p in ps)

            # Associate with item id
            id_to_phonemes[item_id] = expected_phonemes

        assert (
            actual_phonemes == expected_phonemes
        ), f"Got {actual_phonemes}, expected {expected_phonemes} for '{item_text}'"

        print(item_id, "OK")


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

    # -------------
    # compute-stats
    # -------------
    compute_stats_parser = sub_parsers.add_parser(
        "compute-stats", help="Compute audio statistics for dataset(s)"
    )
    compute_stats_parser.add_argument("model", help="Path to model base directory")
    compute_stats_parser.add_argument(
        "--dataset", required=True, help="Path to dataset directory"
    )
    compute_stats_parser.set_defaults(func=do_compute_stats)

    # ---------
    # phonemize
    # ---------
    phonemize_parser = sub_parsers.add_parser(
        "phonemize",
        help="Generate phonemes for text from stdin according to TTS config",
    )
    phonemize_parser.add_argument(
        "text", nargs="*", help="Text to phonemize (default: stdin)"
    )
    phonemize_parser.add_argument(
        "--config", required=True, help="Path to TTS JSON configuration file"
    )
    phonemize_parser.add_argument(
        "--separator",
        default="",
        help="Separator to add between phonemes (default: none)",
    )
    phonemize_parser.set_defaults(func=do_phonemize)

    # ---------------
    # verify-phonemes
    # ---------------
    verify_phonemes_parser = sub_parsers.add_parser(
        "verify-phonemes", help="Path to TTS JSON configuration file"
    )
    verify_phonemes_parser.add_argument(
        "--language", required=True, help="Language for model (e.g. en-us)"
    )
    verify_phonemes_parser.add_argument(
        "--config", required=True, help="Path to TTS JSON configuration file"
    )
    verify_phonemes_parser.set_defaults(func=do_verify_phonemes)

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
    synthesize_parser.add_argument(
        "--output-dir", help="Directory to write output WAV files (default: play)"
    )
    synthesize_parser.add_argument(
        "--output-file", help="Path to write output WAV file"
    )
    synthesize_parser.add_argument(
        "--use-cuda", action="store_true", help="Use GPU (CUDA) for synthesis"
    )
    synthesize_parser.add_argument(
        "--phonemes", action="store_true", help="Text input is phonemes"
    )
    synthesize_parser.add_argument(
        "--phoneme-separator",
        default=" ",
        help="Separator between input phonemes (default: space)",
    )
    synthesize_parser.add_argument(
        "--accent-language", help="Map phonemes from accent language"
    )
    synthesize_parser.set_defaults(func=do_synthesize)

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
    serve_parser.add_argument(
        "--use-cuda", action="store_true", help="Use GPU (CUDA) for synthesis"
    )
    serve_parser.add_argument(
        "--cache-dir", help="Path to directory to cache WAV files (default: no cache)"
    )
    serve_parser.set_defaults(func=do_serve)

    # Shared arguments
    for sub_parser in [
        init_parser,
        compute_stats_parser,
        synthesize_parser,
        serve_parser,
        phonemize_parser,
        verify_phonemes_parser,
    ]:
        sub_parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to console"
        )

    return parser.parse_args()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
