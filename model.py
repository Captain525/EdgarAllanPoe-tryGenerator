#look at this website https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/gpt2#overview
import tensorflow as tf
from preprocess import *
from collectData import collectDataFromTextDoc
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import random
from GPT2 import GPT2
vocabSize = 2048
embedSize = 1000
textCorpus = get_data_lines("data/")
print(textCorpus)
#sample a line from the dataset? 
sample = random.sample(textCorpus, 1)[0]
print(sample)
mergedSample = merge_lines(sample, use_bos = True, order = None)
print(mergedSample)

#print(textCorpus)
print(len(textCorpus))
print("before tokenizer")
special_tokens = {
    #"sep_token": "<LINE>",
    "pad_token": "<PAD>", 
    #"bos_token": "<BOS>", 
    #"eos_token": "<EOS>"

}
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#tokenizer.add_special_tokens(special_tokens)
tokenizedText = tokenizer(mergedSample)['input_ids']
tokenizer.pad_token = tokenizer.eos_token
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
poemList = []
#for poem in textCorpus:

    #poemList.append(merge_lines(poem, True, None))
tokenizedDataDict = tokenizeDataset(poemList, tokenizer, True, False)
tokenizedInput = tokenizedDataDict['input_ids']
tokenizedLabels = tokenizedDataDict['labels']
attentionMask = tokenizedDataDict['attention_mask']
print(tokenizedInput)
#model with weights
print("before model")
print("sequence size: ", tokenizedInput.shape[1])


#model= TFGPT2LMHeadModel.from_pretrained("gpt2")
model = GPT2(50257, tokenizedInput.shape[1], embeddingSize = 768, nLayers = 12, nHead = 12, nInner = 4*768, tokenizer = tokenizer)
print(tokenizedInput[0:5])
#model.build([[None, tokenizedInput.shape[1], attentionMask.shape], [None, tokenizedLabels.shape[1]]])
model.build([tokenizedInput[0:5].shape, None, attentionMask[0:5].shape, None, None, None, None, None, None, None, None, None, None, False, tokenizedLabels[0:5].shape])
model.run_eagerly = True
print("after model")
print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate = .01)
model.compile(optimizer)

print(tokenizedInput.shape, tokenizedLabels.shape)
model.fit((tokenizedInput, attentionMask, tokenizedLabels), y = None, batch_size = 10, epochs = 20)
