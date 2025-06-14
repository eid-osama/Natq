import numpy as np
import torch
import sys
import os
import custom_arabic_to_phones
from nemo.collections.tts.models import HifiGanModel, FastPitchModel,MixerTTSModel
import soundfile as sf
import IPython.display as ipd
import noisereduce as nr
from catt.utils import remove_non_arabic
import gc
# Optional CATT diacritizer
sys.path.append('catt')
from eo_pl import TashkeelModel
from tashkeel_tokenizer import TashkeelTokenizer


def fastpitch_infer_plain_arabic(arabic_text,model, seed=42):
    # Load pretrained models
    if(model == "fastpitch"):
        model_path = "FastPitch--val_loss=0.7130-epoch=49-last.ckpt"
        model = FastPitchModel.load_from_checkpoint(checkpoint_path=model_path).eval().cuda()
    elif(model == "mixer"):
        model_path = "Mixer-TTS--val_mel_loss=0.7356-epoch=57.ckpt"
        model = MixerTTSModel.load_from_checkpoint(checkpoint_path=model_path).eval().cuda()
    hfg_path_gt = "HifiGan--val_loss=0.3817-epoch=34-last.ckpt"

    
    vocoder_model = HifiGanModel.load_from_checkpoint(checkpoint_path=hfg_path_gt).eval().cuda()

    # =====================
    # Step 1: Tashkeel using CATT
    # =====================
    tokenizer = TashkeelTokenizer()
    ckpt_path = 'catt/models/best_eo_mlm_ns_epoch_193.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tashkeel_model = TashkeelModel(tokenizer, max_seq_len=1024, n_layers=6, learnable_pos_emb=False)
    tashkeel_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    tashkeel_model.eval().to(device)
    arabic_text = [arabic_text]
    arabic_text = [remove_non_arabic(i) for i in arabic_text]
    arabic_text = tashkeel_model.do_tashkeel_batch(arabic_text, batch_size=16, verbose=False)

    print("Tashkeel Arabic text:", arabic_text)

    # =====================
    # Step 2: Convert to phonemes
    # =====================
    phoneme_seq = custom_arabic_to_phones.custom_arabic_to_phones(arabic_text[0])
    phoneme_str = " ".join(phoneme_seq)
    print("Phoneme string:", phoneme_str)

    # =====================
    # Step 3: Generate tokens and spectrogram
    # =====================
    with torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

        tokens = model.parse(phoneme_str, normalize=False)
        if tokens.shape[1] < 3:
            print("Warning: input is too short for Conv1D, padding tokens.")
            pad = torch.zeros((tokens.shape[0], 3 - tokens.shape[1]), dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, pad], dim=1)

        spectrogram = model.generate_spectrogram(tokens=tokens)
        audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)

    # =====================
    # Step 4: Normalize and return audio
    # =====================
    audio = audio.to('cpu').numpy()[0]
    audio = audio / np.abs(audio).max()
    torch.cuda.empty_cache()
    gc.collect()
    return audio

# === Inference example ===
# if __name__ == "__main__":
#     arabic_input = "السلام عليكم ورحمة الله وبركاته"
#     audio = fastpitch_infer_plain_arabic(arabic_input,'mixer')
#     sf.write('output.wav', audio, 22050)
#     ipd.display(ipd.Audio('output.wav', autoplay=True))
