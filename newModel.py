import tensorflow as tf
from preprocess import *
from postprocess import Postprocessing

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from generate import *
import random
from GPT2FineTune import GPT2FineTune
from metrics import Metrics

def splitData(data, attentionMask):
    proportion = .8
    
    size = data.shape[0]
    indices = tf.range(0, size, 1)
    shuffledIndices = tf.random.shuffle(indices)
    numTrain = int(proportion*size)
    trainData = tf.gather(data, shuffledIndices[:numTrain])
    trainMask = tf.gather(attentionMask, shuffledIndices[:numTrain])

    validationData =  tf.gather(data, shuffledIndices[numTrain: ])
    validationMask = tf.gather(attentionMask, shuffledIndices[numTrain:])
    return trainData, trainMask, validationData, validationMask
def evaluatePoemGeneration(model, tokenizer):
    min_length = 10
    max_length = 30
    lines = generate_lines(model, tokenizer,min_length, max_length, True, False, ["It was a dark and stormy night",None,  "I hated to hear the sound of my beating heart"], 2, 1, True, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    
    print(lines)
    splitPoems = np.array(breakPoemLines(lines), dtype = str)
    numPoems = len(splitPoems)
    print("split poems: ", splitPoems)
    print(numPoems)
    sum = 0
    for poem in splitPoems:
        lexicalDiversity= evaluate.lexical_diversity(poem.numpy())
        sum+= lexicalDiversity
    avgDiversity = sum/numPoems
    print("lexical diversity: ", avgDiversity)


loadWeights = True
addSpecial = False
text = get_data_poems("data/")
postprocess= Postprocessing()
evaluate = Metrics()

#NO SPECIAL TOKENS. THIS BREAKS EVERYTHING. 
special_tokens = {
    "sep_token": "<LINE>",
    "pad_token": "<PAD>", 
    "bos_token": "<BOS>", 
    "eos_token": "<EOS>"
}

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if loadWeights:
    model = GPT2FineTune(len(tokenizer), "weights/")
else:
    model = GPT2FineTune(len(tokenizer))
if addSpecial:
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
else:
    tokenizer.pad_token = tokenizer.eos_token


poems = mergePoems(text, addSpecial)
print(poems)
encodedText = tokenizeDataset(poems, tokenizer, True, False)
input_ids = encodedText["input_ids"]
print("input i ds: ", input_ids)
attention_mask = encodedText["attention_mask"]

trainData, trainMask, valData, valMask = splitData(input_ids, attention_mask)
#trainData = input_ids[0:100]
#valData = input_ids[0:100]
#trainMask = attention_mask[0:100]
#valMask = attention_mask[0:100]
adam = tf.keras.optimizers.Adam()
model.compile(adam, metrics = [tf.keras.metrics.Mean(name = "perplexity")])
#model.run_eagerly = True
#only put in the ids rn, maybeput in more stuff later, but itll be hard to get it to work. 
evaluatePoemGeneration(model, tokenizer)
tokenized = tf.convert_to_tensor(tokenizer("once upon a midnight dreary")['input_ids'], tf.int32)[tf.newaxis, ...]
print(tokenized)
generated = model.generate(tokenized, 10, 30, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
print(generated)
output =  batch_decode(generated, tokenizer, True, False, False)
print("output text: ", output)

model.fit((trainData[0:100], trainMask[0:100]), epochs = 3, batch_size =2, validation_data = (valData, valMask))
model.save_weights("weights")
evaluatePoemGeneration(model, tokenizer)
