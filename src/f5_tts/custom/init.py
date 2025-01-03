import os
import time
import importlib
import importlib.resources
import re
import tempfile
from typing import Any, Literal, Type

import numpy as np
import soundfile as sf

import torch
import vocos
import vocos.feature_extractors
from huggingface_hub import hf_hub_download
from cached_path import cached_path


from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
from f5_tts.model import CFM
from f5_tts.model.backbones.dit import DiT

def init():
    device     : Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    device_capability: tuple[int, int] = (0,0)
    device_name      : str             = ""
    device_properties: Type[object]    = object

    if device == "cuda":
        device_curr       = torch.cuda.current_device()
        device_capability = torch.cuda.get_device_capability(device_curr)
        device_name       = torch.cuda.get_device_name(device_curr)
        device_properties = torch.cuda.get_device_properties(device_curr)
    else:
        device_curr       = torch.cpu.current_device()

    TEXT_TO_SYNTHESIZE = "Here we generate something just for test."
    OUTPUT_DIR         = "custom/out" # relative to src/f5_tts
    OUTPUT_FILE        = TEXT_TO_SYNTHESIZE.lower().strip().replace(" ", "_").removesuffix(".") + ".wav"
    # NOTE: temp wav files can be made with:
    #   `tempfile.NamedTemporaryFile(delete=True, prefix=TEXT_TO_SYNTHESIZE, suffix=".wav")`
    
    current_dir  = os.getcwd()
    HF_CACHE_DIR = os.path.join(current_dir, ".cache/hf")

    REF_AUDIO_FILE = str(importlib.resources.files("f5_tts").joinpath("custom/basic_ref_en.wav")) # reference audio file
    REF_AUDIO_TEXT = "Some call me nature, others call me mother nature."                         # transcript

    # ===========================================================
    # region         vocoder model
    #
    vocoder_name: Literal["vocos", "bigvgan"] = "vocos"
    vocoder_repo_id     = "charactr/vocos-mel-24khz"
    VOCODER_MODEL_PATH  = hf_hub_download(repo_id=vocoder_repo_id, cache_dir=HF_CACHE_DIR, filename="pytorch_model.bin")
    VOCODER_CONFIG_PATH = hf_hub_download(repo_id=vocoder_repo_id, cache_dir=HF_CACHE_DIR, filename="config.yaml")

    # ===========================================================
    # region         checkpoint
    #
    checkpoint_repo_name = "F5-TTS"
    checkpoint_exp_name  = "F5TTS_Base"
    checkpoint_step      = 1200000
    # TODO: why does infer_cli.py assume .safetensors for vocos and .pt for bigvgan ?
    CHECKPOINT_FILE_PATH = str(cached_path(f"hf://SWivid/{checkpoint_repo_name}/{checkpoint_exp_name}/model_{checkpoint_step}.pt", cache_dir=HF_CACHE_DIR))
    
    print(f"{'DEVICE':<25}"                + f" =  {device}")
    if device == "cuda":
        print(f"{'DEVICE_CURR':<25}"       + f" =  {device_curr}")
        print(f"{'DEVICE_CAPABILITY':<25}" + f" =  {device_capability}")
        print(f"{'DEVICE_NAME':<25}"       + f" =  {device_name}")
        print(f"{'DEVICE_PROPERTIES':<25}" + f" =  {device_properties}")

    print("---")
    
    print(f"{'TEXT_TO_SYNTHESIZE':<25}"    + f" =  {TEXT_TO_SYNTHESIZE}")
    print(f"{'OUTPUT_DIR':<25}"            + f" =  {OUTPUT_DIR}")
    print(f"{'OUTPUT_FILE':<25}"           + f" =  {OUTPUT_FILE}")

    print("---")

    print(f"{'REF_AUDIO_FILE':<25}"        + f" =  {REF_AUDIO_FILE}")
    print(f"{'REF_AUDIO_TEXT':<25}"        + f" =  {REF_AUDIO_TEXT}")
    
    print("---")

    print(f"{'HF_CACHE_DIR':<25}"          + f" =  {HF_CACHE_DIR}")
    print(f"{'VOCODER_MODEL_PATH':<25}"    + f" =  {VOCODER_MODEL_PATH}")
    print(f"{'VOCODER_CONFIG_PATH':<25}"   + f" =  {VOCODER_CONFIG_PATH}")
    print(f"{'CHECKPOINT_FILE_PATH':<25}"  + f" =  {CHECKPOINT_FILE_PATH}")

    print("")

    # ===========================================================
    # region         load vocoder
    #
    match vocoder_name:
        case "vocos":
            state_dict : Any         = torch.load(VOCODER_MODEL_PATH, map_location="cpu", weights_only=True)
            vocoder    : vocos.Vocos = vocos.Vocos.from_hparams(VOCODER_CONFIG_PATH)

            if isinstance(vocoder.feature_extractor, vocos.feature_extractors.EncodecFeatures):
                encodec_parameters = {
                    "feature_extractor.encodec." + key: value
                    for key, value in vocoder.feature_extractor.encodec.state_dict().items()
                }
                state_dict.update(encodec_parameters)
            
            vocoder.load_state_dict(state_dict)
            vocoder = vocoder.eval().to(device)

        # TODO:
        # case "bigvgan":
        #     try:
        #         from third_party.BigVGAN import bigvgan
        #     except ImportError:
        #         print("You need to follow the README to init submodule and change the BigVGAN source code.")
        #     if is_local:
        #         """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
        #         vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        #     else:
        #         local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
        #         vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        #     vocoder.remove_weight_norm()
        #     vocoder = vocoder.eval().to(device)
    assert(vocoder_name == "vocos")
    #
    # ----------------------------------------------------------------------------------------------------------------------------------


    # ===========================================================
    # region         load TTS model ( from infer_cli.py )
    #


    # ===========================================================
    # global vars found in utils_infer.py ( #TODO find their usage )
    #
    target_sample_rate  = 24000
    n_mel_channels      = 100
    hop_length          = 256
    win_length          = 1024
    n_fft               = 1024

    ode_method: Literal["euler", "midpoint"] = "euler"

    # vocab.txt file if `tokenizer` is "custom"
    tokenizer                  = "custom"
    vocab_file                 = str(importlib.resources.files("f5_tts").joinpath("custom/vocab.txt"))
    vocab_char_map, vocab_size = init_tokenizer(vocab_file, tokenizer)
    
    # model config.model.arch ( F5TTS_Base_train.yaml for example )
    # is passed to `load_model` from `utils_infer.py` which further passes cfg to `DiT` constructor
    model_cfg = {
        "dim"                    : 1024, #        
        "depth"                  : 22,   #      
        "heads"                  : 16,   #      
        "ff_mult"                : 2,    #     matches `F5TTS_Base_train.yaml`    
        "text_dim"               : 512,  #       
        "conv_layers"            : 4,    #     
        "checkpoint_activations" : False #        
    }
    transformer : torch.nn.Module = DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels)

    #
    # TTS model
    #
    tts_model = CFM(
        transformer=transformer,
        mel_spec_kwargs=dict(
            n_fft              = n_fft,
            hop_length         = hop_length,
            win_length         = win_length,
            n_mel_channels     = n_mel_channels,
            target_sample_rate = target_sample_rate,
            mel_spec_type      = vocoder_name,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)
    
    model_dtype = torch.float16
    if vocoder_name == "bigvgan":
        model_dtype = torch.float32

    tts_model = tts_model.to(model_dtype)
    #
    # ----------------------------------------------------------------------------------------------------------------------------------


    # ===========================================================
    # region         load checkpoint
    #
    checkpoint = torch.load(CHECKPOINT_FILE_PATH, map_location=device, weights_only=True)

    use_ema = True   # use exponential moving average model weights ( from readme 'The `use_ema = True` is harmful for early-stage finetuned checkpoints (which goes just few updates, thus ema weights still dominated by pretrained ones), try turn it off and see if provide better results.'  ' Try turn off use_ema if using an early-stage finetuned checkpoint (which goes just few updates).')

    if use_ema:
        checkpoint["model_state_dict"] = {                         # 
            k.replace("ema_model.", ""): v                         # TODO: replace this mess
            for k, v in checkpoint["ema_model_state_dict"].items() # 
            if k not in ["initted", "step"]                        # 
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

    tts_model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    tts_model = tts_model.to(device)
    #
    # ----------------------------------------------------------------------------------------------------------------------------------

    return tts_model

#
#
#
def init_tokenizer(vocab_file: str, tokenizer: Literal["char", "byte", "custom"] = "byte") -> tuple[dict[str, int], int]:
    """
        tokenizer   - "char" for char-wise tokenizer, need .txt vocab_file
                    - "byte" for utf-8 tokenizer
                    - "custom" if you're directly passing in a path to the vocab.txt you want to use

        vocab_size  - if use "char", derived from unfiltered character & symbol counts of custom dataset
                    - if use "byte", set to 256 (unicode byte range)
    """
    vocab_char_map = {}
    vocab_size     = 256

    match tokenizer:
        case "char":
            with open(vocab_file, "r", encoding="utf-8") as f:
                for i, char in enumerate(f):
                    vocab_char_map[char[:-1]] = i
            assert vocab_char_map[" "] == 0, "first line of vocab.txt must be empty (used for unknown char)"
            vocab_size = len(vocab_char_map)

        case "byte":
            vocab_size = 256

        case "custom":
            with open(vocab_file, "r", encoding="utf-8") as f:
                for i, char in enumerate(f):
                    vocab_char_map[char[:-1]] = i
            vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size