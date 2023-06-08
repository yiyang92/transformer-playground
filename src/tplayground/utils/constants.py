from enum import Enum


class TransformerType(Enum):
    encoder_based = "encoder_based"
    decoder_based = "decoder_based"
    enc_dec_based = "enc_dec_based"

    def __str__(self):
        return self.value
