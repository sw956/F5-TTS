import importlib
import re
import io
import tempfile
from typing import Literal

import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text


app = FastAPI()

class SynthesizePayloadMelo(BaseModel):
    text: str = 'Ahoy there matey! There she blows!'
    language: str = 'EN'
    speaker: str = 'EN-US'
    speed: float = 1.0

class SynthesizePayload_F5(BaseModel):
    text                : str
    voice               : str
    tts_model           : str             = "F5-TTS"

    target_sample_rate  : int             = 24000
    save_chunk          : bool            = False   # save each audio chunks during inference
    remove_silence      : bool            = False   # remove long silence found in ouput

    target_rms          : float           = 0.1     # target output speech loudness normalization value
    cross_fade_duration : float           = 0.15    # duration of cross-fade between audio segments in seconds
    nfe_step            : Literal[16, 32] = 32      # number of function evaluation (denoising steps)
    cfg_strength        : float           = 1.0     # classifier-free guidance strength  TODO: what default value ( 1.0 / 2.0 ) ?
    sway_sampling_coef  : float           = -1.0    #
    speed               : float           =  1.0    #
    fix_duration        : float | None    = None    # fix the total duration (ref and gen audios) in seconds


def synth(payload: SynthesizePayload_F5):
    # TODO:
    DEFAULT_VOICE = payload.voice
    voices = {}
    # DEFAULT_VOICE = "main"
    # voices = {
    #     "main": {
    #         "ref_audio": ref_audio_file,
    #         "ref_text" : ref_audio_text
    #     }
    # }
    # voice = voices[DEFAULT_VOICE]
    # print("Voice:", voice)
    # voice["ref_audio"], voice["ref_text"] = preprocess_ref_audio_text(voice["ref_audio"], voice["ref_text"]) # TODO: extract fn
    # print("ref_audio" , voice["ref_audio"], "\n\n")

    generated_audio_segments = []

    """
        EXAMPLE:

        `gen_text = "Hello [narrator] this is [character_A] a test."`
        
        OUTPUT:

        chunk: "Hello "      voice: DEFAULT_VOICE  (no match)
        chunk: " this is "   voice: "narrator"
        chunk: " a test."    voice: "character_A"
    """
    reg1   = r"(?=\[\w+\])"       # matches [words_like_this] ( \w represents [a-zA-Z0-9_] )
    reg2   = r"\[(\w+)\]"         # captures e.g. "words_like_this" from "[words_like_this]"
    chunks = re.split(reg1, payload.text)

    for chunk in chunks:
        if not chunk.strip():
            continue

        match = re.match(reg2, chunk)   
        if match:
            voice = match[1]
        else:
            voice = DEFAULT_VOICE

        if voice not in voices:
            print(f"Voice {voice} not found, using {DEFAULT_VOICE}.")
            voice = DEFAULT_VOICE
        chunk = re.sub(reg2, "", chunk)
        
        ref_audio_chunk = voices[voice]["ref_audio"]
        ref_text_chunk  = voices[voice]["ref_text"]
        gen_text_chunk  = chunk.strip()

        #
        # SYNTHESIZE CALL
        #
        # TODO: extract
        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio_chunk,
            ref_text_chunk,
            gen_text_chunk,
            model_obj           ="F5-TTS",
            vocoder             = vocoder,
            mel_spec_type       = vocoder_name,
            target_rms          = payload.target_rms,
            cross_fade_duration = payload.cross_fade_duration,
            nfe_step            = payload.nfe_step,
            cfg_strength        = payload.cfg_strength,
            sway_sampling_coef  = payload.sway_sampling_coef,
            speed               = payload.speed,
            fix_duration        = payload.fix_duration,
            device              = "cuda",
        )
        generated_audio_segments.append(audio_segment)


    # TODO:
    #     if save_chunk:
    #         if len(gen_text_) > 200:
    #             gen_text_ = gen_text_[:200] + " ... "
    #         sf.write(
    #             os.path.join(output_chunk_dir, f"{len(generated_audio_segments)-1}_{gen_text_}.wav"),
    #             audio_segment,
    #             final_sample_rate,
    #         )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
        return final_wave

    return


@app.post("/stream")
async def synthesize_stream(payload: SynthesizePayload_F5):
    language = payload.language
    text = payload.text
    speaker = payload.speaker or list(models[language].hps.data.spk2id.keys())[0]
    speed = payload.speed

    def audio_stream():
        bio = io.BytesIO()
        # models[language].tts_to_file(text, models[language].hps.data.spk2id[speaker], bio, speed=speed, format='wav')
        audio_data = bio.getvalue()
        yield audio_data

    return StreamingResponse(audio_stream(), media_type="audio/wav")