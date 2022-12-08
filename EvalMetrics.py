from spacy.lang.en import English
from spacy.tokenizer import tokenizer
import string as string_utils
import pronouncing
from collections import Counter

def rhymingDistance(poem):
    """
    Generally poe uses pairs of rhymes I think, two consecutive lines rhyme. 
    input: Poem as a list of lines, line as list of words.  maybe not line as list of words not sure. 
    """
    nlp = English()
    tokenizer = nlp.tokenizer
    #this stuff gets rid of punctuation to make the rhymes work. 
    for idx, line in enumerate(poem):
        while len(line) >= 1 and line[-1] in string_utils.punctuation:
            line = line[:-1]
        poem[idx] = line
    for line in poem:
        if line == "":
            print("EMPTY LINE ERROR")
            return None
    numLines = len(poem)
    #get last word from each line. 
    listLastWords = []
    for line in poem:
        lastWord = line[-1]
        #listLastWords.append(lastWord)
        print("last word tokenized: ", tokenizer(line)[-1].text)
        listLastWords.append(tokenizer(line)[-1].text)
    distance = 0
   
    pairs = [(i,i+1) for i in range(numLines-1)]
    for pair in pairs:
        firstWord = listLastWords[pair[0]]
        secondWord = listLastWords[pair[1]]
        phones_0 = pronouncing.phones_for_word(firstWord)
        if phones_0 == []:
            return None
        phones_0 = pronouncing.rhyming_part(phones_0[0])
        phones_1 = pronouncing.phones_for_word(secondWord)
        if phones_1 == []:
            return None
        phones_1 = pronouncing.rhyming_part(phones_1[0])
        #if they have a different rhyme add one to the distance. 
        if phones_0 != phones_1:
            distance += 1 / len(pairs)
    return distance

        
def calculateWordFreq(poems):
    nlp = English()
    tokenizer = nlp.tokenizer
    oedilf_word_freq = Counter()
    #COUNT OCCURENCES OF WORDS IN THE POEMS. 
    for poem in poems: 
        for line in poem:
            words = [token.text for token in tokenizer(line)]
            oedilf_word_freq.update(words)
    #DONT COUNT PUNCTUATION AS WORDS. 
    for punct in string_utils.punctuation:
        if punct in oedilf_word_freq:
            oedilf_word_freq.pop(punct)

    return oedilf_word_freq
def getWordFreq(files):
    generated_word_freq = Counter()

    for filename in files:
        with open(filename, 'r') as file:
            for _ in range(100):
                poem = []
                for _ in range(5):
                    poem.append(file.readline().strip())
                file.readline()
                for line in poem:

                    words = [token.text for token in tokenizer(line)]
                    generated_word_freq.update(words)

    for punct in string_utils.punctuation:
        if punct in generated_word_freq:
            generated_word_freq.pop(punct)

    return generated_word_freq

def get_coverage(oedilf_word_freq, generated_word_freq, min_word_freq):
    top_words = set()
    for word, count in oedilf_word_freq.most_common():
        if count < min_word_freq:
            break
        top_words.add(word)

    covered, total = 0, 0
    for word, count in generated_word_freq.most_common():
        if word in top_words:
            covered += count
        total += count    

    coverage = covered / total
    return coverage
