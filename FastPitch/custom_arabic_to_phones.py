import sys
sys.path.append('tts-arabic-pytorch/text/')  # Add the script's directory to Python path
from phonetise_buckwalter import (
    arabic_to_buckwalter,
    buckwalter_to_arabic,
    process_utterance
)

def custom_arabic_to_phones(text, return_phonemes=True):
    bw_text = arabic_to_buckwalter(text)
    if return_phonemes:

        phoneme_str = process_utterance(bw_text).replace("+ ", "").strip()

        phoneme_list = phoneme_str.split()
        return phoneme_list
    return bw_text