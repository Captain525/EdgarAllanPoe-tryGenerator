import numpy as np
import os
import string

def get_data(path='../data'):
    
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
