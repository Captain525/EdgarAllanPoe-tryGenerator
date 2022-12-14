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
def countUnique(poem):
    """
    Poem as list of words. 
    """
    poemArray = np.array(poem)
    uniqueWords = np.unique(poemArray)
    lexDiv = uniqueWords.shape[0]/poemArray.shape[0]
    return lexDiv
def evaluatePoemGeneration(model, tokenizer):
    min_length = 10
    max_length = 50
    lines = generate_lines(model, tokenizer,min_length, max_length, True, False, [None, "Once upon a midnight dreary", "The heart beat beneath the floorboards"], 5, 1, True, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    for line in lines:
        print(line + "\n")
    print(lines)
   
    splitPoems = breakPoemLines(lines)
    numPoems = len(splitPoems)
    print("split poems: ", splitPoems)
    print(numPoems)
    sum = 0
    for poem in splitPoems:
        lexicalDiversity= countUnique(poem)
        sum+= lexicalDiversity
    avgDiversity = sum/numPoems
    print("lexical diversity: ", avgDiversity)

def runModel(epochs, learningRate, modelNum):
    """
    Main method called when you want to run the model. 
    """
    #whether to use presaved weights or train new ones. 
    loadWeights = False
    #whether to add special tokens to the text. 
    addSpecial = False
    #get poems in list form. 
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
    #make tokenizer. 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if addSpecial:
        tokenizer.add_special_tokens(special_tokens)
    else:
        tokenizer.pad_token = tokenizer.eos_token
    if loadWeights:
        model = GPT2FineTune(len(tokenizer), "weights/")
    else:
        model = GPT2FineTune(len(tokenizer),None,  modelNum)



    #merge each poem into one long string. 
    poems = mergePoems(text, addSpecial)
    #convert poems into tokens. 
    encodedText = tokenizeDataset(poems, tokenizer, False, False)
    input_ids = encodedText["input_ids"]
    attention_mask = encodedText["attention_mask"]
    trainData, trainMask, valData, valMask = splitData(input_ids, attention_mask)

    adam = tf.keras.optimizers.Adam(learning_rate = learningRate)
    model.compile(adam, metrics = [tf.keras.metrics.Mean(name = "perplexity")])
    #model.run_eagerly = True
    #only put in the ids rn, maybeput in more stuff later, but itll be hard to get it to work.
    if loadWeights:
        evaluatePoemGeneration(model, tokenizer)
    else:
        #train. 
        model.fit((trainData, trainMask), epochs = epochs, batch_size =2, validation_data = (valData, valMask))
        model.save_weights("weights")
        evaluatePoemGeneration(model, tokenizer)
runModel(10, 5e-3, 0)