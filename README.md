# Larynx

A [fork](https://github.com/rhasspy/TTS) of [MozillaTTS](https://github.com/mozilla/TTS) that uses [gruut](https://github.com/rhasspy/gruut) for cleaning and phonemizing text.

Will be used by the [Rhasspy project](https://github.com/rhasspy) to train freely available voices from public datasets.

See the [tutorial](docs/tutorial.md) below for step by step instructions.

## Dependencies

* Python 3.7 or higher
* [PyTorch](https://pytorch.org/) >= 1.5
* [gruut](https://github.com/rhasspy/gruut)
* [rhasspy/TTS](https://github.com/rhasspy/TTS) (MozillaTTS fork, `dev` branch)

## Differences from MozillaTTS

MozillaTTS (maintained by the awesome [erogol](https://github.com/erogol)) models are typically trained on a set of [phonemes](https://en.wikipedia.org/wiki/Phoneme) derived from text for a given language. The [phonemizer](https://github.com/bootphon/phonemizer) tool is used by default, which calls out to [espeak-ng](https://github.com/espeak-ng/) to guess phonemes for words.

Inside MozillaTTS, there is a file called [symbols.py](https://github.com/rhasspy/TTS/blob/dev/TTS/tts/utils/text/symbols.py) that contains a large set (129) of phonemes meant to cover a large number of languages:

```python
# Phonemes definition
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilic
```

Contrast this with the set of phonemes (45) used by [gruut](https://github.com/rhasspy/gruut) for U.S. English.

```text
_ | ‖ # aɪ aʊ b d d͡ʒ eɪ f h i iː j k l m n oʊ p s t t͡ʃ uː v w z æ ð ŋ ɑ ɑː ɔ ɔɪ ə ɛ ɝ ɡ ɪ ɹ ʃ ʊ ʌ ʒ θ
```

Fewer phonemes means smaller models, which means faster training and synthesis.

Larynx is intended to be used on small datasets from volunteers, typically with only 1,000 examples. We therefore do more work upfront, making it so the model does not have to learn about [dipthongs](https://en.wikipedia.org/wiki/Diphthong), short/long vowels, or all ways of writing [breaks](https://en.wikipedia.org/wiki/Prosodic_unit).

## Datasets

Larynx assumes your datasets follow a simple convention:

* A `metadata.csv` file
    * Delimiter is `|` and there is no header
    * Each row is `id|text`
    * Each corresponding WAV file must be named `<id>.wav`
* WAV files in the same directory
    * All WAVs have the same sample rate (22050 recommended)

## Installation

See `scripts/create-venv.sh`

This includes cloning [rhasspy/TTS](https://github.com/rhasspy/TTS) as a submodule (`dev` branch).

## Usage

Before training, you must initialize a model directory. Larynx will scan your dataset(s) and generate appropriate config files for both TTS and a vocoder.

### Initialization

```sh
$ python3 -m larynx init /path/to/model --language <LANGUAGE> --dataset /path/to/dataset
```

Add `--model-type glowtts` for GlowTTS instead of Tactron2.

See `python3 -m larynx init --help` for more options.

### Training (Tacotron2)

```sh
$ python3 TTS/TTS/bin/train_tts.py \
    --config /path/to/model/config.json
```

### Training (GlowTTS)

You should have added `--model-type glowtts` during initialization.

```sh
$ python3 TTS/TTS/bin/train_glow_tts.py \
    --config /path/to/model/config.json
```

### Training (Vocoder)

```sh
$ python3 TTS/TTS/bin/train_vocoder.py \
    --config /path/to/model/vocoder/config.json
```

### Synthesis

```sh
$ python3 -m larynx synthesis \
    --model /path/to/model/<timestamp>/best_model.pth.tar \
    --config /path/to/model/<timestamp>/config.json \
    --vocoder-model /path/to/model/vocoder/<timestamp>/best_model.pth.tar \
    --vocoder-config /path/to/model/vocoder/<timestamp>/config.json \
    --output-file /path/to/test.wav \
    'This is a test sentence!'
```

If you have [sox](http://sox.sourceforge.net/) installed, you can leave off `--output-file` and type lines via standard in. They will be played using the `play` command.

You may also specify `--output-dir` to have each sentence (line on stdin or argument) written to a different WAV file.
