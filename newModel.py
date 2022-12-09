import tensorflow as tf
from preprocess import *
from collectData import collectDataFromTextDoc
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import random
from GPT2FineTune import GPT2FineTune


text = get_data_poems("data/")
#print("text corpus: ", text[0:10])
#NO SPECIAL TOKENS. THIS BREAKS EVERYTHING. 
special_tokens = {
    "sep_token": "<LINE>",
    "pad_token": "<PAD>", 
    "bos_token": "<BOS>", 
    "eos_token": "<EOS>"
}
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(len(tokenizer))
config = GPT2Config()
model = GPT2FineTune(len(tokenizer))
#tokenizer.add_special_tokens(special_tokens)
#resize the embedding layers to adjust for this new thing. 
#model.resize_token_embeddings(len(tokenizer))
poems = mergePoems(text)
encodedText = tokenizeDataset(poems, tokenizer, True, False)
#print("encoded Text variable: ", encodedText)
adam = tf.keras.optimizers.Adam()
model.compile(adam)
#dataset = tf.data.Dataset.from_tensor_slices((encodedText['input_ids'], encodedText['attention_mask'], encodedText['labels']))
inputDict = {"input_ids": encodedText['input_ids'], "attention_mask":encodedText['attention_mask'], "labels":encodedText['labels']}
#only put in the ids rn, maybeput in more stuff later, but itll be hard to get it to work. 
#model.fit(encodedText['input_ids'][0:1000], epochs = 1, batch_size = 2)
encodedPrompt = tf.convert_to_tensor(tokenizer.encode("It was a dark and stormy night"), dtype = tf.int32)[ tf.newaxis, ...]
print(encodedPrompt)
generatedText = model.generate(encodedPrompt, max_length = 20)[0]
print("generated text encoded: ", generatedText)
decoded = tokenizer.decode(generatedText)
print("decoded text: ", decoded)

