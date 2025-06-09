import sys
sys.path.append('tts-arabic-pytorch')
import nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers as tts_tokenizers
import text as txt

class ArabicPhonemeTokenizer(tts_tokenizers.BaseTokenizer):

    def __init__(self):
        self.phonemes = txt.symbols
        # Create phoneme dictionary
        self.phoneme_dict = txt.phon_to_id_
        self.vocab = self.phoneme_dict
        self.itos = list(self.vocab.keys())  # Index-to-string mapping
        self.stoi = self.vocab  # String-to-index mapping
        # self.pad = txt.phon_to_id_["pad"]
        super().__init__(tokens=self.phoneme_dict)

    def encode(self, text):
        tokens = txt.phonemes_to_tokens(text, append_space=False)
        return txt.tokens_to_ids(tokens)

    def decode(self, tokens):
        return ' '.join(txt.ids_to_tokens(tokens))
