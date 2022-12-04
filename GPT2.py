from transformers import TFGPT2Model, TFGPT2LMHeadModel
import tensorflow as tf
class GPT2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.pretrained = TFGPT2LMHeadModel.from_pretrained('gpt2')
        self.summary()

    def summary(self):
        for layer in self.pretrained.layers:
            print(layer)
        