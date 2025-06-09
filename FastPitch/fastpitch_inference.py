import numpy as np
import torch
import arabic_phoneme_tokenizer
from nemo.collections.tts.models import HifiGanModel, FastPitchModel
import IPython.display as ipd
import soundfile as sf
import os
import sys
sys.path.append('tts-arabic-pytorch')
import custom_arabic_to_phones
import mishkal.tashkeel

def fastpitch_infer_plain_arabic(arabic_text, seed=20):

    fastpitch_model_path = "FastPitch--val_loss=0.5327-epoch=79-last.ckpt"  
    hfg_path_gt = "HifiGan--val_loss=0.3817-epoch=34-last.ckpt"
    vocoder_model = HifiGanModel.load_from_checkpoint(checkpoint_path=hfg_path_gt).eval().cuda()
    fastpitch_model = FastPitchModel.load_from_checkpoint(checkpoint_path=fastpitch_model_path).eval().cuda()

    # vocalizer = mishkal.tashkeel.TashkeelClass()
    # tashkeel_arabic_text= vocalizer.tashkeel(arabic_text)
    # print("Tashkeel Arabic text:", tashkeel_arabic_text)
    # 1. Convert plain Arabic text to phoneme sequence
    phoneme_seq = custom_arabic_to_phones.custom_arabic_to_phones(arabic_text)
    print("Phoneme sequence:", phoneme_seq)
    # Join tokens to a space-separated string if needed
    phoneme_str = " ".join(phoneme_seq)
    print("Phoneme string:", phoneme_str)

    with torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

        # Tokenize phoneme string using your tokenizer
        tokens = fastpitch_model.parse(str_input=phoneme_str, normalize=False)
        print("Tokens:", tokens)
        # Generate spectrogram
        spectrogram = fastpitch_model.generate_spectrogram(tokens=tokens)
        # Vocoder: spectrogram to audio
        audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)

    # Convert to numpy array and normalize
    audio = audio.to('cpu').numpy()[0]
    audio = audio / np.abs(audio).max()
    return audio

audio = fastpitch_infer_plain_arabic("هَذِهِ لَقْطَةٌ مُكَثَّفَةٌ وَحَزينَةٌ مُقَطَّعَةٌ مِنْ شَريطِ طالِبِ عِلْمَ قَدَّرِ اللَّهِ عَلَيْهُ أَنْ يَعيشَ لَحْظَةَ التَّكْوِينِ العِلْميِّ فِي عَصْرِ ثَوْرَةِ نُظُمِ الِاتِّصَالَاتِ وَشَبَكاتِ التَّواصُل",seed=42)
sf.write('output.wav', audio, 22050)
ipd.display(ipd.Audio('output.wav', autoplay=True))