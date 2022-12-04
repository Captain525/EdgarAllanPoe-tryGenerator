from transformers import TFGPT2Model, TFGPT2LMHeadModel
import tensorflow as tf
class GPT2(tf.keras.Model):
    def __init__(self):
        self.pretrained = TFGPT2LMHeadModel.from_pretrained('gpt2')
        for layer in self.pretrained.layers:
            print(layer.summary())
        