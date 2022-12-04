#look at this website https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/gpt2#overview
import tensorflow as tf
from preprocess import *
from collectData import collectDataFromTextDoc
from transformers import TFGPT2Model, GPT2Tokenizer
import random
vocabSize = 2048
embedSize = 1000
textCorpus = get_data_poems("data/")
#sample a line from the dataset? 
sample = random.sample(textCorpus, 1)[0]
print(sample)
mergedSample = merge_lines(sample, use_bos = True, order = None)
print(mergedSample)

#print(textCorpus)
print(len(textCorpus))
print("before tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens = {
    "sep_token": "<LINE>",
    "pad_token": "<PAD>", 
    "bos_token": "<BOS>", 
    "eos_token": "<EOS>"

}
tokenizer.add_special_tokens(special_tokens)
tokenizedText = tokenizer(mergedSample)['input_ids']
untokenized  = tokenizer.decode(tokenizedText)

detokenizePart = tokenizer.decode(tokenizedText[0:2])


#print(untokenized)
#print(tokenizedText)

#print(tokenizedText)
#linesReversed = reverseLineOrder(mergedSample.split(" "), use_bos = True, tokenizer = tokenizer)
#print(linesReversed)
print("tokenized length: ", len(tokenizedText))
input_ids = list(reverseLineOrder(tokenizedText, use_bos = True, tokenizer = tokenizer, reverse_last_line = True))
#print(input_ids)
print("reversed length: ", len(input_ids))
detokenizePartOfInput = tokenizer.decode(tokenizedText[0:2])
print("detokenized: ", detokenizePartOfInput)
decodeString = tokenizer.decode(input_ids, clean_up_tokenization_spaces = True)

print(decodeString)
#configuration = GPT2Config(vocab_size = vocabSize, n_positions = 2048, n_embd= embedSize, n_layer = 12, n_head = 12, n_inner = None, activation_function = "relu")

#model with weights
print("before model")
model = TFGPT2Model.from_pretrained('gpt2')
print(model.summary())
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer)
model.fit(tokenizedText)