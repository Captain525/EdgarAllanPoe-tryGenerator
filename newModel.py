import tensorflow as tf
from preprocess import *
from collectData import collectDataFromTextDoc
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
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
model.fit(encodedText['input_ids'], epochs = 3, batch_size = 2)

