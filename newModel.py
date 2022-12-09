import tensorflow as tf
from preprocess import *

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from generate import *
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
print("got to fit")
model.fit((encodedText['input_ids'][0:100], encodedText['attention_mask'][0:100]), epochs = 1, batch_size = 2)
model.save_weights("weights")
listPhrases = ["It was a dark and stormy night", "I hated to hear the sound of my beating heart"]

encodedPrompt = tokenizeDataset(listPhrases, tokenizer, True, True)

#promptBatch = tf.concat([encodedPrompt, encodedPrompt2], axis=0)
print("encodedPRompt: ", encodedPrompt)
#output = model(encodedPrompt)
generatedText = model.generate(encodedPrompt['input_ids'],max_length = 30, bos_token_id = tokenizer.bos_token_id, eos_token_id = tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id,)

print("generated text encoded: ", generatedText)

decoded =batch_decode(generatedText, tokenizer, True, True, False)
print("decoded text: ", decoded)
generatedNoPrompt = model.generate(None, max_length = 20, bos_token_id = tokenizer.bos_token_id, eos_token_id = tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id)


