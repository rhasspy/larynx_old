# Tutorial

In this tutorial, we'll be training a Larynx model from the [Kathleen](https://github.com/rhasspy/dataset-voice-kathleen/) dataset, a U.S. English female voice. Before starting, make sure you have the [necessary materials](#materials-needed).

There are 5 distinct stages we'll cover:

1. [Installing Larynx](#installing-larynx)
2. [Prepare the data](#data-preparation)
3. [Train the voice](#train-voice)
4. [Train the vocoder](#train-vocoder)
5. [Try it out](#synthesis)

Stages 3 and 4 can be done simultaneously if you have more than one GPU.

## Materials Needed

To train any Larynx voice, you need:

* Python 3.7 or higher
* [CUDA](https://developer.nvidia.com/cuda-zone) and [cuDNN](https://developer.nvidia.com/cudnn)
    * Tested on Ubuntu 18.04 (bionic) with CUDA 10.2 and cuDNN 7.6
* A CUDA-enabled GPU that's supported by [PyTorch](https://pytorch.org/) 1.5 or higher
    * Larynx was developed with a [GTX 1060 6GB](https://www.nvidia.com/en-in/geforce/products/10series/geforce-gtx-1060/)
    * Unsupported GPUs can sometimes be used by [compiling PyTorch from source](https://github.com/pytorch/pytorch#from-source)
* [Sox](http://sox.sourceforge.net/)
    * Usually just `apt-get install sox`

You will probably also want to install:

* [GNU Parallel](https://www.gnu.org/software/parallel/)
    * `apt-get install parallel`
* [jq](https://stedolan.github.io/jq/)
    * `apt-get install jq`
    
## Installing Larynx

To start, clone the [Larynx repo](https://github.com/rhasspy/larynx) and its [MozillaTTS fork](https://github.com/rhasspy/TTS):

```sh
$ git clone --recursive https://github.com/rhasspy/larynx.git
$ cd larynx
```

Next, create a virtual environment and update it:

```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip3 install --upgrade pip
$ pip3 install --upgrade wheel setuptools
```

Finally, set up the MozillaTTS submodule and install all dependencies:

```sh
$ pushd TTS
$ pip3 install -r requirements.txt
$ python3 setup.py develop
$ popd
$ pip3 install -r requirements.txt
```

If all is well, you should be able to execute the `bin/larynx` script.

## Data Preparation

Larynx assumes datasets live in a single directory with:

* A `metadata.csv` file with `id|text` rows (no header)
* WAV files named `<id>.wav` for each row in `metadata.csv`
    * Typically 22050 Hz 16-bit mono PCM
    
We'll start by downloading the [Kathleen dataset](https://github.com/rhasspy/dataset-voice-kathleen/):

```sh
$ mkdir -p local
$ git clone https://github.com/rhasspy/dataset-voice-kathleen.git local/kathleen
```

This dataset has WAV files with transcription text files next to them:

```sh
$ ls local/kathleen/data/ | head
arctic_a0001_1592748574.txt
arctic_a0001_1592748574.wav
arctic_a0001_1592748585.txt
arctic_a0001_1592748585.wav
arctic_a0001_1592748600.txt
arctic_a0001_1592748600.wav
arctic_a0002_1592748530.txt
arctic_a0002_1592748530.wav
arctic_a0002_1592748544.txt
arctic_a0002_1592748544.wav
```

The WAV files are also 48 Khz stereo, so we'll need to convert:

```sh
$ soxi local/kathleen/data/arctic_a0001_1592748574.wav

Input File     : 'local/kathleen/data/arctic_a0001_1592748574.wav'
Channels       : 2
Sample Rate    : 48000
Precision      : 16-bit
Duration       : 00:00:03.30 = 158230 samples ~ 247.234 CDDA sectors
File Size      : 633k
Bit Rate       : 1.54M
Sample Encoding: 16-bit Signed Integer PCM
```

Let's start by gathering the transcripts into a single file in `local/kathleen/larynx`:

```sh
$ mkdir -p local/kathleen/larynx
$ truncate -s 0 local/kathleen/larynx/metadata.csv
$ find local/kathleen/data -type f -name '*.txt' | \
    while read -r fname; do \
      id="$(basename "${fname}" .txt)"; \
      text="$(cat "${fname}")"; \
      printf '%s|%s\n' "${id}" "${text}" >> local/kathleen/larynx/metadata.csv; \
    done
```

Check that everything went OK:

```sh
$ head local/kathleen/larynx/metadata.csv
arctic_b0204_1592536717|Down through the perfume weighted air fluttered the snowy fluffs of the cottonwoods.
arctic_a0286_1592710508|Give them their choice between a fine or an official whipping.
arctic_a0199_1592712782|Thus had the raw wilderness prepared him for this day.
arctic_b0128_1592698769|Philip bent low over Pierre.
arctic_b0025_1592700841|Now, you understand.
arctic_b0397_1592530241|The hunters were still arguing and roaring like some semi-human amphibious breed.
arctic_a0417_1592706106|It is also an insidious, deceitful sun.
arctic_b0439_1592529025|The land exchanged its austere robes for the garb of a smiling wanton.
arctic_b0213_1592536528|Some boy, she laughed acquiescence.
arctic_a0002_1592748530|Not at this particular case, Tom, apologized Whittemore.
```

Now convert the WAV files using `sox` and [GNU Parallel](https://www.gnu.org/software/parallel/):

```sh
$ find local/kathleen/data -type f -name '*.wav' -print0 | \
    parallel -0 sox {} -r 22050 -c 1 -e signed-integer -b 16 -t wav local/kathleen/larynx/{/}
```

Check that there are the same number of WAV files:

```sh
$ find local/kathleen/data -type f -name '*.wav' | wc -l
1620

$ find local/kathleen/larynx -type f -name '*.wav' | wc -l
1620
```

and that the converted files have the right format:

```sh
$ soxi local/kathleen/larynx/arctic_a0001_1592748574.wav

Input File     : 'local/kathleen/larynx/arctic_a0001_1592748574.wav'
Channels       : 1
Sample Rate    : 22050
Precision      : 16-bit
Duration       : 00:00:03.30 = 72687 samples ~ 247.235 CDDA sectors
File Size      : 145k
Bit Rate       : 353k
Sample Encoding: 16-bit Signed Integer PCM
```

Looks good!

## Train Voice

Once the data has been prepared, we can initialize our training directory.
This assumes you are in the `larynx` base directory with the virtual environment active.

```sh
$ bin/larynx init local/kathleen/train \
    --language en-us \
    --name kathleen \
    --model-type glowtts \
    --dataset local/kathleen/larynx \
    --debug
```

After some time, you should have the following files in `local/kathleen/train`:

* `config.json` - MozillaTTS config file for voice
* `dataset_phonemes.csv` - phonemized sentences from all datasets
* `guessed_words.txt` - words whose pronunciations had to be guessed
* `phonemes.txt` - model phonemes with their index
* `phoneme_cache/` - directory with pre-computed phonemes for all datasets
* `scale_stats.npy` - audio statistics for all datasets
* `vocoder/`
    * `config.json` - MozillaTTS config file for vocoder
    
It's very important to check `guessed_words.txt`, since `gruut` is having to guess how some words are pronounced:

```sh
$ head local/kathleen/train/guessed_words.txt
head local/kathleen/train/guessed_words.txt
pearce's p ɪ ɹ s ɪ z
hanrahan's h æ n ɹ ə h æ n z
provocateurs p ɹ oʊ v ɑː k ə t ɝ z
daylight's d eɪ l aɪ t s
springy s p ɹ ɪ ŋ i
nightglow n aɪ t ɡ l oʊ
kerfoot's k ɝ f ʊ t s
factor's f æ k t ɝ z
thorpe's θ ɔ ɹ p s
jeanne's d͡ʒ iː n z
```

The first column is the word, and the following symbols are from the [International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet). You can hear how they're supposed to sound by visiting [IPA Chart](https://www.ipachart.com/).

Beware of **numbers** especially. `gruut` will expand numbers into words automatically, but this can sometimes lead to unexpected results. For example, the year 1999 will not be expanded as expected:

```sh
$ echo '1999' | gruut en-us tokenize | jq -r .clean_text
one thousand nine hundred and ninety nine
```

If you need to re-run `larynx init`, you can add `--skip-audio-stats` to avoid re-computing the audio statistics.

### Run the Training Script

We're ready to actually start training! Simply run:

```sh
$ python3 TTS/TTS/bin/train_glow_tts.py \
    --config_path local/kathleen/train/config.json
```

This should begin training a voice in a directory inside `local/kathleen/train/model`.

Note that we used the `train_glow_tts.py` script because we're using a GlowTTS model. If you use a Tacotron2 model, run `train.py` instead.

To see how your model is doing during training, open a second terminal and install [tensorboard](https://www.tensorflow.org/tensorboard/):

```sh
$ cd larynx
$ source .venv/bin/activate
$ pip3 install tensorboard
```

Once its installed, run it with:

```sh
$ tensorboard --logdir local/kathleen/train/model/kathleen-<timestamp>
```

where `<timestamp>` is the date/time of the run.

Visit http://localhost:6006 to see the graphs and listen to test audio as it comes out! You usually have to wait for the first few steps to complete before anything will show up.

## Train Vocoder

A vocoder helps your voice sound far less robotic. Training the vocoder is almost identical to training the voice:

```sh
$ python3 TTS/TTS/bin/train_vocoder.py \
    --config_path local/kathleen/train/vocoder/config.json
```

To avoid running out of GPU memory, I've had to reduce the `batch_size` in the vocoder `config.json` to 25. You can do this manually or pass `--vocoder-batch-size` to `larynx init`.

Vocoder training takes a long time, but it appears you can re-use a previous trained vocoder if the audio parameters are all the same (see the `audio` section of `config.json`). By using `--continue_path` instead of `--config_path` with `train_vocoder.py`, it's possible to tune a vocoder for a new voice. The pitfalls and complexities of this are beyond the scope of this tutorial. Just know that any difference in the `audio` parameters between your vocoder and your voice configs usually means you need to re-train from scratch.

## Synthesis

To test out your voice, use `larynx synthesize`:

```sh
$ bin/larynx synthesize 'Welcome to the world of speech synthesis!' \
    --model /path/to/best_model.pth.tar
```

This assumes that `config.json` is in the same directory as `best_model.pth.tar`.

You should hear the sentence spoken (using `play`) and have it exit. Passing no text arguments will read lines from standard in, allowing you hear things interactively. The `--output-dir` and `--output-file` arguments control how WAV files are saved.

If your vocoder is also trained:

```sh
$ bin/larynx synthesize 'Welcome to the world of speech synthesis!' \
    --model /path/to/best_model.pth.tar \
    --vocoder-model /path/to/vocoder/best_model.pth.tar
```

This assumes that the vocoder's `config.json` is in the same directory as `vocoder/best_model.pth.tar`.

### Web Server

Finally, you can launch a TTS web server for your voice:

```sh
$ bin/larynx serve \
    --model /path/to/best_model.pth.tar \
    --vocoder-model /path/to/vocoder/best_model.pth.tar
```

Visit http://localhost:5002 to try it out.
