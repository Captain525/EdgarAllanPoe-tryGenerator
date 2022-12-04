#look at this website https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/gpt2#overview
import tensorflow as tf
from preprocess import get_data
from collectData import collectDataFromTextDoc
from transformers import TFGPT2Model, GPT2Tokenizer
vocabSize = 2048
embedSize = 1000
textCorpus = get_data("data/")
print(len(textCorpus[0]))
print("before tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizedText = tokenizer(textCorpus, return_length = True)
print(tokenizedText.length)
#print(tokenizedText[0])
#configuration = GPT2Config(vocab_size = vocabSize, n_positions = 2048, n_embd= embedSize, n_layer = 12, n_head = 12, n_inner = None, activation_function = "relu")

#model with weights
print("before model")
model = TFGPT2Model.from_pretrained('gpt2')
print(model.summary())
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer)
model.fit(tokenizedText)