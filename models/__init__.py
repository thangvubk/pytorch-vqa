from image_models import *
from vqa_models import *

def get_vqa_mode(name, vocab_size):
    if name is 'san':
        return SAN(vocab_size)
    elif name is 'saa':
        return SAA(vocab_size)
    elif name is 'proposed':
        return Proposed(vocab_size)

def get_image_model(name):
    pass #TODO
