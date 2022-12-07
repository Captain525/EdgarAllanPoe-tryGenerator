import numpy as np
import os
import string
import tensorflow as tf
from datasets import Dataset


def get_data(path='../data'):
    """
    Gets a list of strings, where each string is a word in a line. 
    """
    
    poem_lst = []
    files = os.listdir(path)
    for file in files: # read in all poems
        poem_path = os.path.join(path, file)
        with open(poem_path, 'r') as poem:
            poem_text = poem.read()
        poem_lst.append(poem_text.lower().translate(str.maketrans('', '', string.punctuation)))
    
    output_lst = []
    for poem in poem_lst:
        poem = poem.replace("\n", " <LINE> ")
        poem = "<BOS> " + poem + " <EOS>"
        split_poem = poem.split(" ")
        while ("" in split_poem):
            split_poem.remove("")
        output_lst.append(split_poem)

    return output_lst

def get_data_lines(path = '../data'):
    """
    Gets a list of lines, where each line is a string. 
    """
    poem_lst = []
    files = os.listdir(path)
    for file in files: # read in all poems
        poem_path = os.path.join(path, file)
        with open(poem_path, 'r') as poem:
            poem_text = poem.readlines()
        poem_lst.append(poem_text)
    
    output_lst = []
    for poem in poem_lst:
       
        poem[0] = "<BOS> " + poem[0]
        poem[-1]= poem[-1] + " <EOS>"
        for line in poem:
            line = line.replace("\n", "").lower()
            #print(repr(line))
            if line == "":
                continue
            if("\n" in line):
                assert(1==0)
            output_lst.append(line)
    return output_lst
def splitPoem(poemText):
    """
    Split a poem which is too large into smaller chunks. 
    """
    #listPoemParts = poemText.split(".")
    listIfPeriod = ["." in line for line in poemText]
    indicesTrueList = [0] + [i+1 for i,value in enumerate(listIfPeriod) if value == True]
    listSubPoems = []
    for i in range(0, len(indicesTrueList)-1):
        index = indicesTrueList[i]
        nextIndex = indicesTrueList[i+1]
       
        subList = poemText[index:nextIndex]
        if(len(subList) == 0):
            print("zero length")
            continue
        listSubPoems.append(subList)
    endPortion = poemText[indicesTrueList[-1]:]
    listSubPoems.append(endPortion)
    sumSubListLengths = sum(len(subpoem) for subpoem in listSubPoems) 
    print("sum sublist lengths: ", sumSubListLengths)
    print("poem text length: ", len(poemText))
    print(listSubPoems)
    print(poemText)
    assert(sumSubListLengths == len(poemText))
 
    #want lines after the poem part to split into separate poems. 
    poemList = []
    for poem in listSubPoems:
        newLength = len(merge_lines(poem, True, None))
        if(newLength>1024):
            #just don't include it in the final result. 

            print("TOO BIG: ", newLength )
            continue
        else:
            poemList.append(poem)
    return poemList

def get_data_poems(path = '.../data'):
    """
    Gets data as a list of poems, where each poem is 
    a  list of lines, where each line is a string. 
    """
    poem_lst = []
    files = os.listdir(path)
    for file in files: # read in all poems
        poem_path = os.path.join(path, file)
        with open(poem_path, 'r') as poem:
            poem_text = poem.readlines()
            if(len(merge_lines(poem_text, True, None))>1024):
                listParts = splitPoem(poem_text)
                poem_lst = poem_lst + listParts
            else:
                poem_lst.append(poem_text)
    
    output_lst = []
    for poem in poem_lst:
        lineList = []
        #will add BOS and EOS later. 
        for id, line in enumerate(poem):
            line = line.replace("\n", "").lower()
            #print(repr(line))
            if line == "":
                continue
            #gets rid of ending punctuation in line. 
            if line[-1] in string.punctuation:
                line = line[:-1]
            assert("\n" not in line)
            lineList.append(line)
        output_lst.append(lineList)
    return output_lst

def merge_lines(lines, use_bos, order = None):
    """
    Input is a list of "line" strings. 
    Output makes it one string but adds line delimiters. 
    """
    #reorders the lines based on the input list/ordering. 
    if order is not None: 
        try: 
            order = list(order)
        except Exception:
            return
        assert isinstance(order, list)
        assert(sorted(order) == list(range(len(order))))
        lines = [lines[0] for o in order]
    #merges the lines into one. 
    words = ' <LINE> '.join(lines) + ' <LINE>'
    #adds bos if we need to. 
    if(use_bos):
        words = '<BOS> ' + words + ' <EOS>'
     
    words = ' '.join(words.split())
    return words
def reorder(lines, order = None):
    if order is None:
        return lines
    else: 
        new = [(o,i) for i,o in enumerate(order)]
        #sort by o not by i. the value of o which is 0 is in the first index now, and its coupled with teh i it corresponds to. 
        new = sorted(new)
        #get the enumerate values from the sorted list. These represent the index of the input which should be in each spot. 
        new = [o[1] for o in new]
        #if new[0] = 1, that means the 2nd line of lines had the lowest element of the order, which means that it should be first in the output. 
        #switch order of lines based on those enumerate values. 
        lines = [lines[o] for o in new]
    return lines
def reverseLineOrder(input_ids, use_bos, tokenizer, reverse_last_line = False):
    pad_token_id = tokenizer.pad_token_id
    print(pad_token_id)
    start = 0
    #basically get the starting point of the sequence without padding. 
    for i, id_ in enumerate(input_ids):
        if id_ !=pad_token_id:
            init, start = i,i
            break
    tmp_input_ids = input_ids[start:]
    new_input_ids = np.zeros_like(tmp_input_ids, dtype = np.int32)

    if use_bos:
        #transmit the BOS string to the start of the reverse order sentence. 
        new_input_ids[0] = tmp_input_ids[0]
        start = 1

    else:
        start = 0
    print("sep token: ", tokenizer.sep_token_id)
    for end in range(1, len(tmp_input_ids)):
        
        #iterate through sentence until you reach a sep token. 
        if tmp_input_ids[end] == tokenizer.sep_token_id:
            #make all the ids in that area reversed, ie you reverse the line. 
            #print(tmp_input_ids[start:end])
            new_input_ids[start:end] = tmp_input_ids[start:end][::-1]
            #add the sep token at the end. 
            new_input_ids[end] = tokenizer.sep_token_id
            #move on to later words/phrases. 
        
            #print(new_input_ids[start:end+1])
            #print(new_input_ids[start:end+1].shape)
            start = end + 1
    #shoudl end up with start at the beginning of the last line.
    # If we want to reverse it, we can reverse entire ending from start to end.  
    if reverse_last_line:
        #new input ids from start to hte end
        new_input_ids[start:-1] = tmp_input_ids[start:-1][::-1]
        #EOS CHARACTER. NOT IN ORIGINAL CODE, SO MAY BE WRONG. 
        new_input_ids[-1] = tmp_input_ids[-1]
    else:
        new_input_ids[start:] = tmp_input_ids[start:]
    new_input_ids = np.concatenate([input_ids[:init], new_input_ids], axis=0)
    return new_input_ids.astype(np.int32)

def tokenizeDataset( batch, tokenizer, use_bos, reverse):
    """
    Takes in a batch of the dataset, then 
    """
    if not reverse:
        
        
        batch = tokenizer(batch, padding = "max_length", max_length = 1024, return_tensors = "tf")
    else:
        batch = tokenizer(batch, padding = "max_length", max_length = 1024, return_tensors = "np")
        for i, input_ids in enumerate(batch['input_ids']):
            batch['input_ids'][i] = reverseLineOrder(batch['input_ids'][i], use_bos = use_bos, tokenizer = tokenizer)
        batch['input_ids'] = tf.convert_to_tensor(batch['input_ids'])
        batch['attention_mask'] = tf.convert_to_tensor(batch['attention_mask'])
    #pytorch code has clone(batch.detach())#
    #check this is right. 
    #identity meant to make a copy, stop gradients meant to replicate detach. 
    batch['labels'] = tf.identity(tf.stop_gradient(batch['input_ids']))
    return batch

