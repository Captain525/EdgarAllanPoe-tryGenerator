from preprocess import get_data
import numpy as np
import random

class RandomGenerator:
    """
    Makes random decisions for words, generates words randomly from vocabList. 
    """
    def __init__(self, line_range, num_lines):
        # line range is a list of two elements, min number of words per line, and max (can be the same number)
        self.line_min = line_range[0]
        self.line_max = line_range[1]
        self.constant_lines = (self.line_min == self.line_max)
        self.num_lines = num_lines
        data = get_data()
        single_data_lst = [word for poem in data for word in poem]
        self.vocab = np.unique(np.array(single_data_lst)).tolist() # entire vocab
        
    def generate_poem(self):
        poem = ["<BOS>"] # start of poem token
        
        for _ in range(self.num_lines): # iterate through number of lines
            line_length = None
            
            if self.constant_lines: # get line lenght
                line_length = self.line_min
            else:
                line_length = random.randint(self.line_min, self.line_max)
            
            for _ in range(line_length): # pick random words from vocab for line length
                word = self.vocab[random.randint(0, len(self.vocab)-1)]
                poem.append(word)
            
            poem.append("<LINE>") # add line token to end of the line
        
        poem.pop()
        poem.append("<EOS>") # end of poem token
        
        return poem
        