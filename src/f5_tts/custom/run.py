import importlib

import importlib.resources
from typing import Literal

import vocos.feature_extractors

from f5_tts.model.backbones.dit import DiT
from f5_tts.model import CFM

import torch
import vocos


#
#
#
def load_tokenizer(vocab_file: str, tokenizer: Literal["char", "byte", "custom"] = "byte") -> tuple[dict[str, int], int]:
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


#
# main entrypoint for `infer_cli.py`
#
def main():
    # main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    # if "voices" not in config:
    #     voices = {"main": main_voice}
    # else:
    #     voices = config["voices"]
    #     voices["main"] = main_voice
    # for voice in voices:
    #     print("Voice:", voice)
    #     print("ref_audio ", voices[voice]["ref_audio"])
    #     voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
    #         voices[voice]["ref_audio"], voices[voice]["ref_text"]
    #     )
    #     print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    # generated_audio_segments = []
    # reg1 = r"(?=\[\w+\])"
    # chunks = re.split(reg1, gen_text)
    # reg2 = r"\[(\w+)\]"
    # for text in chunks:
    #     if not text.strip():
    #         continue
    #     match = re.match(reg2, text)
    #     if match:
    #         voice = match[1]
    #     else:
    #         print("No voice tag found, using main.")
    #         voice = "main"
    #     if voice not in voices:
    #         print(f"Voice {voice} not found, using main.")
    #         voice = "main"
    #     text = re.sub(reg2, "", text)
    #     ref_audio_ = voices[voice]["ref_audio"]
    #     ref_text_ = voices[voice]["ref_text"]
    #     gen_text_ = text.strip()
    #     print(f"Voice: {voice}")
    #     audio_segment, final_sample_rate, spectragram = infer_process(                       # NOTE: `infer_process` calls `infer_batch_process` which does the actual inference
    #         ref_audio_,
    #         ref_text_,
    #         gen_text_,
    #         ema_model,
    #         vocoder,
    #         mel_spec_type=vocoder_name,
    #         target_rms=target_rms,
    #         cross_fade_duration=cross_fade_duration,
    #         nfe_step=nfe_step,
    #         cfg_strength=cfg_strength,
    #         sway_sampling_coef=sway_sampling_coef,
    #         speed=speed,
    #         fix_duration=fix_duration,
    #     )
    #     generated_audio_segments.append(audio_segment)

    #     if save_chunk:
    #         if len(gen_text_) > 200:
    #             gen_text_ = gen_text_[:200] + " ... "
    #         sf.write(
    #             os.path.join(output_chunk_dir, f"{len(generated_audio_segments)-1}_{gen_text_}.wav"),
    #             audio_segment,
    #             final_sample_rate,
    #         )

    # if generated_audio_segments:
    #     final_wave = np.concatenate(generated_audio_segments)

    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     with open(wave_path, "wb") as f:
    #         sf.write(f.name, final_wave, final_sample_rate)
    #         # Remove silence
    #         if remove_silence:
    #             remove_silence_for_generated_wav(f.name)
    #         print(f.name)
    return

def load_vocoder(device, vocoder_name, is_local=False, local_path="", hf_cache_dir=None):

    return

#
#
#
#
#
if __name__ == "__main__":
    model_name : str                    = "F5-TTS"
    device     : Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    gen_text = "Here we generate something just for test."           # text to synthesize
    gen_file = ""

    output_dir  = ""
    output_file = ""
    
    VOCODER_PATH    = "" # TODO: see https://huggingface.co/SWivid/F5-TTS`
    CHECKPOINT_PATH = "" # TODO: see https://huggingface.co/SWivid/F5-TTS`

    ref_audio = "infer/examples/basic/basic_ref_en.wav"              # reference audio file
    ref_text  = "Some call me nature, others call me mother nature." # transcript for ref audio file


    save_chunk              = False  # save each audio chunks during inference
    remove_silence          = False  # remove long silence found in ouput
    load_vocoder_from_local = True   # load vocoder from local dir

    target_rms          : float                        = 0.1     # target output speech loudness normalization value
    cross_fade_duration : float                        = 0.15    # duration of cross-fade between audio segments in seconds
    nfe_step            : Literal[16, 32]              = 32      # number of function evaluation (denoising steps)
    cfg_strength        : float                        = 1.0     # classifier-free guidance strength  TODO: what default value ( 1.0 / 2.0 ) ?
    ode_method          : Literal["euler", "midpoint"] = "euler" #
    sway_sampling_coef  : float                        = -1.0    #
    speed               : float                        =  1.0    #
    fix_duration        : float | None                 = None    # fix the total duration (ref and gen audios) in seconds  TODO: wtf ( replace `None` code with `0.0f` in 'utils_infer.py` )

    #
    # global vars found in utils_infer.py ( #TODO find their usage )
    #
    target_sample_rate  = 24000
    n_mel_channels      = 100
    hop_length          = 256
    win_length          = 1024
    n_fft               = 1024


    #
    # from infer_cli.py # TODO
    #
    """
    # patches for pip pkg user
    if "infer/examples/" in ref_audio:
        ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
    if "infer/examples/" in gen_file:
        gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
    if "voices" in config:
        for voice in config["voices"]:
            voice_ref_audio = config["voices"][voice]["ref_audio"]
            if "infer/examples/" in voice_ref_audio:
                config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))

    # ignore gen_text if gen_file provided
    if gen_file:
        gen_text = codecs.open(gen_file, "r", "utf-8").read()

    # output path
    wave_path = Path(output_dir) / output_file
    # spectrogram_path = Path(output_dir) / "infer_cli_out.png"
    if save_chunk:
        output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
        if not os.path.exists(output_chunk_dir):
            os.makedirs(output_chunk_dir)
    """



    #
    # load vocoder
    #
    vocoder_name: Literal["vocos", "bigvgan"] = "vocos"
    # TODO: download vocoder from hf:
    # repo_id     = "charactr/vocos-mel-24khz"
    # config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
    # model_path  = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
    match vocoder_name:
        case "vocos":
            config_path = ""           # f"{local_path}/config.yaml"
            model_path  = VOCODER_PATH # f"{local_path}/pytorch_model.bin"
            state_dict  = torch.load(model_path, map_location="cpu", weights_only=True)
            
            vocoder : vocos.Vocos = vocos.Vocos.from_hparams(config_path)

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
    #
    # ----------------------------------------------------------------------------------------------------------------------------------


    #
    # load TTS model ( from infer_cli.py )
    #

    # vocab.txt file if `tokenizer` is "custom"
    tokenizer                  = "custom" # TODO
    vocab_file                 = ""       # TODO
    vocab_char_map, vocab_size = load_tokenizer(vocab_file, tokenizer)
    
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
    model = CFM(
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

    model = model.to(model_dtype)
    #
    # ----------------------------------------------------------------------------------------------------------------------------------


    #
    # load checkpoint
    #
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

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

    model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    model = model.to(device)
    #
    # ----------------------------------------------------------------------------------------------------------------------------------

    main() # TODO