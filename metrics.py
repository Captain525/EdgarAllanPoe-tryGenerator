import numpy as np
import string
import os

class Metrics:
    def __init__(self):
        None
    
    def lexical_diversity(self, formatted_poem): # formatted poem from postproccess file
        # remove all tokens and compute TTR
        poem_arr = formatted_poem
        #poem_arr = np.delete(formatted_poem, np.where(formatted_poem == "\n"))
        #poem_arr = np.delete(poem_arr, np.where(poem_arr == "<BOS>"))
        #poem_arr = np.delete(poem_arr, np.where(poem_arr == "<EOS>"))
        return len(np.unique(poem_arr)) / len(poem_arr)
