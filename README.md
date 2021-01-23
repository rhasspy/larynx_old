# Larynx

![Larynx logo](docs/img/logo.png)

A [fork](https://github.com/rhasspy/TTS) of [MozillaTTS](https://github.com/mozilla/TTS) that uses [gruut](https://github.com/rhasspy/gruut) for cleaning and phonemizing text.

Used by the [Rhasspy project](https://github.com/rhasspy) to train freely available voices from public datasets.
See [pre-trained models](#pre-trained-models).

See the [tutorial](docs/tutorial.md) below for step by step instructions.

Once installed, you can run a [web server](#web-server) and test it out at http://localhost:5002

## Dependencies

* Python 3.7 or higher
* [PyTorch](https://pytorch.org/) >= 1.6
* [gruut](https://github.com/rhasspy/gruut)
* [rhasspy/TTS](https://github.com/rhasspy/TTS) (MozillaTTS fork, `dev` branch)

## Pre-Trained Models

Models and Docker images are available here:

* Dutch
    * [nl_larynx-rdh](https://github.com/rhasspy/nl_larynx-rdh)
* German
    * [de_larynx-thorsten](https://github.com/rhasspy/de_larynx-thorsten)
* French
    * [fr_larynx-siwis](https://github.com/rhasspy/fr_larynx-siwis)
* Spanish
    * [es_larynx-css10](https://github.com/rhasspy/es_larynx-css10)
* Russian
    * [ru_larynx-nikolaev](https://github.com/rhasspy/ru_larynx-nikolaev)

If you use Home Assistant, these are also available as [Hass.io add-ons](https://github.com/rhasspy/hassio-addons/)

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

Fewer phonemes means smaller models, which means faster training and synthesis. Unfortunately, this means **Larynx models are not compatible with vanilla MozillaTTS.**

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

You need to activate the virtual env via `source .venv/bin/activate` and you can leave it via `deactivate`.

### Docker

A CPU-only Docker image is available at `rhasspy/larynx` with no voices included. See the [voices](#voices) section for Docker images containing specific voices.

```sh
$ docker run -it -p 5002:5002 \
    --device /dev/snd:/dev/snd \
    rhasspy/larynx:<VOICE>-<VERSION>
```

See [web server](#web-server) section for endpoints.

You can leave off `--device` if you don't plan to play test audio through your speakers.

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
    --config_path /path/to/model/config.json
```

### Training (GlowTTS)

You should have added `--model-type glowtts` during initialization.

```sh
$ python3 TTS/TTS/bin/train_glow_tts.py \
    --config_path /path/to/model/config.json
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

### Web Server

Run a web server at http://localhost:5002

```sh
$ python3 -m larynx serve \
    --model /path/to/model/<timestamp>/best_model.pth.tar \
    --config /path/to/model/<timestamp>/config.json \
    --vocoder-model /path/to/model/vocoder/<timestamp>/best_model.pth.tar \
    --vocoder-config /path/to/model/vocoder/<timestamp>/config.json \
    --cache-dir /tmp/larynx
```

Endpoints:

* `/api/tts` - returns WAV audio for text
    * `GET` with `?text=...`
    * `POST` with text body
* `/api/phonemize` - returns phonemes for text
    * `GET` with `?text=...`
    * `POST` with text body
* `/process` - compatibility endpoint to emulate [MaryTTS](http://mary.dfki.de/)
    * `GET` with `?INPUT_TEXT=...`
