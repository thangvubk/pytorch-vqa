from image_models import *
from vqa_models import *

def get_vqa_model(name, vocab_size):
    if name == 'san':
        return SAN(vocab_size)
    elif name == 'isan':
        return ISAN(vocab_size)

