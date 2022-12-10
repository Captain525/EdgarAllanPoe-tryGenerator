# libraries
import numpy as np
import string
import os

# class to postproccess the outputted poems to format them, print them, and/or save them to a provided file given a path.
class Postprocessing:
    def __init__(self, path=None, model_name=None):
        self.num_saves = 1
        self.path = path
        self.model_name = model_name
    
    def format_poem(self, poem_arr):
        # take out tokens
        poem_arr[poem_arr == "<LINE>"] = "\n"
        poem_arr = np.delete(poem_arr, np.where(poem_arr == "<BOS>"))
        poem_arr = np.delete(poem_arr, np.where(poem_arr == "<EOS>"))
        return poem_arr

    def print_poem(self, formatted_poem_arr):
        # convert to list and print joined str
        poem_lst = formatted_poem_arr.tolist()
        print(" ".join(poem_lst))
        return None

    def save_poem(self, formatted_poem_arr):
        # open file and find end of file
        save_file = open(self.path, "w")
        save_file.seek(0, 2)
        
        # write header
        header = "\n\n" + self.model_name + "Output Number" + f": {self.num_saves}\n"
        save_file.write(header)
        
        # write peom
        poem_str = " ".join(formatted_poem_arr.tolist())
        save_file.write(poem_str)
        
        # close file and update num_saves
        save_file.close()
        self.num_saves = self.num_saves + 1
        return None
    
    def full_postprocess(self, poem_arr):
        # call all above functions
        formatted_poem = self.format_poem(poem_arr)
        self.print_poem(formatted_poem)
        self.save_poem(poem_arr)
        return None
    