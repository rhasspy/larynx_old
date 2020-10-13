#!/usr/bin/env python3
import io
import json
import os
import time

import torch

from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.utils.generic_utils import setup_generator


def tts(
    model,
    vocoder_model,
    text,
    CONFIG,
    use_cuda,
    ap,
    use_gl,
    speaker_fileid,
    speaker_embedding=None,
    gst_style=None,
    text_is_phonemes=False,
):
    t_1 = time.time()
    waveform, _, _, mel_postnet_spec, _, _ = synthesis(
        model=model,
        text=text,
        CONFIG=CONFIG,
        use_cuda=use_cuda,
        ap=ap,
        speaker_id=speaker_fileid,
        style_wav=gst_style,
        truncated=False,
        enable_eos_bos_chars=CONFIG.enable_eos_bos_chars,
        use_griffin_lim=use_gl,
        speaker_embedding=speaker_embedding,
        backend="torch",
        do_trim_silence=False,
        text_is_phonemes=text_is_phonemes,
    )

    if CONFIG.model == "Tacotron" and not use_gl:
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T).T

    if not use_gl:
        waveform = vocoder_model.inference(
            torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0)
        )

    if use_cuda and not use_gl:
        waveform = waveform.cpu()

    if not use_gl:
        waveform = waveform.numpy()

    waveform = waveform.squeeze()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps))
    return waveform


class Synthesizer:
    def __init__(
        self,
        config_path,
        model_path,
        use_cuda=False,
        vocoder_path="",
        vocoder_config_path="",
        batched_vocoder=True,
        speakers_json="",
        speaker_fileid=None,
        gst_style=None,
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.vocoder_path = vocoder_path
        self.vocoder_config_path = vocoder_config_path
        self.batched_vocoder = batched_vocoder
        self.speakers_json = speakers_json
        self.speaker_fileid = speaker_fileid
        self.gst_style = gst_style

        self.model = None

    def load(self):
        # load the config
        C = load_config(self.config_path)
        self.config = C

        stats_path = C.audio.get("stats_path")
        if stats_path and not os.path.isfile(stats_path):
            # Look for stats next to config
            model_stats_path = os.path.join(
                os.path.dirname(self.config_path), "scale_stats.npy"
            )
            if os.path.isfile(model_stats_path):
                # Patch config
                C.audio["stats_path"] = model_stats_path

        C.forward_attn_mask = True

        # load the audio processor
        ap = AudioProcessor(**C.audio)
        self.ap = ap

        # if the vocabulary was passed, replace the default
        if "characters" in C.keys():
            symbols, phonemes = make_symbols(**C.characters)
        else:
            from TTS.tts.utils.text.symbols import phonemes, symbols

        speaker_embedding = None
        speaker_embedding_dim = None
        num_speakers = 0

        # load speakers
        if self.speakers_json != "":
            speaker_mapping = json.load(open(self.speakers_json, "r"))
            num_speakers = len(speaker_mapping)
            if C.use_external_speaker_embedding_file:
                if self.speaker_fileid is not None:
                    speaker_embedding = speaker_mapping[self.speaker_fileid][
                        "embedding"
                    ]
                else:  # if speaker_fileid is not specificated use the first sample in speakers.json
                    speaker_embedding = speaker_mapping[
                        list(speaker_mapping.keys())[0]
                    ]["embedding"]
                speaker_embedding_dim = len(speaker_embedding)

        self.speaker_embedding = speaker_embedding

        # load the model
        num_chars = len(phonemes) if C.use_phonemes else len(symbols)
        model = setup_model(num_chars, num_speakers, C, speaker_embedding_dim)
        cp = torch.load(self.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(cp["model"])
        model.eval()
        if self.use_cuda:
            model.cuda()

        if hasattr(model.decoder, "set_r"):
            model.decoder.set_r(cp["r"])

        self.model = model

        # load vocoder model
        if self.vocoder_path:
            VC = load_config(self.vocoder_config_path)
            vocoder_model = setup_generator(VC)
            vocoder_model.load_state_dict(
                torch.load(self.vocoder_path, map_location="cpu")["model"]
            )
            vocoder_model.remove_weight_norm()
            if self.use_cuda:
                vocoder_model.cuda()
            vocoder_model.eval()
        else:
            vocoder_model = None
            VC = None

        self.vocoder_model = vocoder_model
        self.vocoder_config = VC

        # synthesize voice
        self.use_griffin_lim = self.vocoder_model is None

        if not C.use_external_speaker_embedding_file:
            if self.speaker_fileid and self.speaker_fileid.isdigit():
                self.speaker_fileid = int(self.speaker_fileid)
            else:
                self.speaker_fileid = None
        else:
            self.speaker_fileid = None

        if (self.gst_style is None) and ("gst" in C.keys()):
            gst_style = C.gst.get("gst_style_input", None)
        else:
            # check if gst_style string is a dict, if is dict convert  else use string
            try:
                gst_style = json.loads(self.gst_style)
                if max(map(int, gst_style.keys())) >= C.gst["gst_style_tokens"]:
                    raise RuntimeError(
                        "The highest value of the gst_style dictionary key must be less than the number of GST Tokens, \n Highest dictionary key value: {} \n Number of GST tokens: {}".format(
                            max(map(int, gst_style.keys())), C.gst["gst_style_tokens"]
                        )
                    )
            except ValueError:
                gst_style = self.gst_style

        self.gst_style = gst_style

    def synthesize(self, text: str, text_is_phonemes: bool = False) -> bytes:
        """Synthesize WAV bytes from text"""
        if not self.model:
            self.load()

        wav = tts(
            self.model,
            self.vocoder_model,
            text,
            self.config,
            self.use_cuda,
            self.ap,
            self.use_griffin_lim,
            self.speaker_fileid,
            speaker_embedding=self.speaker_embedding,
            gst_style=self.gst_style,
            text_is_phonemes=text_is_phonemes,
        )

        with io.BytesIO() as wav_io:
            self.ap.save_wav(wav, wav_io)

            return wav_io.getvalue()
