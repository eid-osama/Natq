from argparse import Namespace
import torch
from SparkTTS.cli.inference import run_tts
from pydub.generators import Sine
from pydub import AudioSegment
import sys
import gc
sys.path.append('catt')
from eo_pl import TashkeelModel
# from ed_pl import TashkeelModel
from tashkeel_tokenizer import TashkeelTokenizer


def is_haraka(char):
    # Unicode range for Arabic diacritics
    return '\u064B' <= char <= '\u0652' or '\u0618' <= char <= '\u061A' or char in ['ٰ', '\u0653', '\u0654', '\u0655']

def normalize_harakat(text):
    # allowed_harakat = ['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ' , 'ا']
    allowed_list = ['ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', 'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ']

    # Map strange or non-standard harakat to close equivalents
    replacement_map = {
        'ٰ': 'َ',
        'ّٰ': 'ّ',
        '\u0618': 'َ',
        '\u0619': 'ُ',
        '\u061A': 'ِ',
        '\u0653': 'َ',
        '\u0654': 'ء',
        '\u0655': 'ء',
        'ۡ': 'ْ',
        'ٱ': 'ا'
    }

    result = ''
    for char in text:
        if char in allowed_list:
            result += char
        elif char in replacement_map:
            replacement = replacement_map[char]
            if replacement in allowed_list:
                result += replacement
            else:
                result += ''  # mapped but not a haraka
        elif is_haraka(char):
            continue  # skip haraka not in allowed list
        else:
            result += char  # keep non-harakat (letters/punctuation/etc.)
    return result

def spark_inference(arabic_text, tashkeel_model, tokenizer):
    # save_dir = "E:\\Natq\\Spark\\output"
    # modified_text =  arabic_text
    # tokenizer = TashkeelTokenizer()
    # ckpt_path = 'catt/models/best_eo_mlm_ns_epoch_193.pt'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    arabic_text = [arabic_text]
    arabic_text = tashkeel_model.do_tashkeel_batch(arabic_text, batch_size=16, verbose=False)
    arabic_text = normalize_harakat(arabic_text[0])
    print("Tashkeel Arabic text:", arabic_text)

    args = Namespace(
        model_dir="Spark-TTS-finetune/pretrained_models/Spark-TTS-0.5B",
        save_dir="E:\\Natq\\Spark\\output",
        device=0,
        text=arabic_text,
        gender='male',
        pitch='moderate',
        speed='moderate',
    )
    run_tts(args)
    torch.cuda.empty_cache()
    gc.collect()
    return True

# text = spark_inference("أهلا وسهلا بكم فى عالمى")
