import torch
import sys
import yaml
import numpy as np
import mishkal.tashkeel
from FastSpeech2.utils.model import get_model, get_vocoder
from FastSpeech2.utils.tools import to_device, synth_samples
from FastSpeech2.text import text_to_sequence
import argparse
sys.path.append('FastSpeech2')
import custom_arabic_to_phones

# This script is used to perform inference using the FastSpeech2 model for Arabic text-to-speech synthesis.
def fastspeech2_infer(
    text,
    restore_step=1500000,
    speaker_id=0,
    pitch_control=1.0,
    energy_control=1.0,
    duration_control=1.0,
    bw=True,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set paths for configs
    preprocess_config_path="FastSpeech2/config/Arabic/preprocess.yaml"
    model_config_path="FastSpeech2/config/Arabic/model.yaml"
    train_config_path="FastSpeech2/config/Arabic/train.yaml"

    # Read configs
    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model and vocoder
    model = get_model(
        argparse.Namespace(
            restore_step=restore_step,
            mode="single",
            bw=bw,
            speaker_id=speaker_id
        ),
        configs,
        device,
        train=False
    )
    vocoder = get_vocoder(model_config, device)

    vocalizer = mishkal.tashkeel.TashkeelClass()
    tashkeel_arabic_text= vocalizer.tashkeel(text)
    print("Tashkeel Arabic text:", tashkeel_arabic_text)
    # 1. Convert plain Arabic text to phoneme sequence
    phoneme_seq = custom_arabic_to_phones.custom_arabic_to_phones(tashkeel_arabic_text)
    print("Phoneme sequence:", phoneme_seq)
    # Join tokens to a space-separated string if needed
    phoneme_str = " ".join(phoneme_seq)
    phonemes = f"{{{phoneme_str}}}"
    print("Phoneme string:", phonemes)

    # Preprocess text
    sequence = np.array(
        #TO_DO
        text_to_sequence(
            phonemes, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    print("The sequance is:  ",np.array(sequence))

    texts = np.array([sequence])
    ids = raw_texts = [text[:100]]
    speakers = np.array([speaker_id])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = (pitch_control, energy_control, duration_control)

    # Synthesize and collect output
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            # synth_samples saves audio to file, but you can modify it to return audio
            # For now, let's assume you want to save to file as before:
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )
    return True

# fastspeech2_infer(
#     text="أريد أن أذهب إلى المدرسة",
# )
