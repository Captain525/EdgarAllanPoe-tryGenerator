import os
def collectDataFromTextDoc():
    folderPath = "data/"
    poemList = []
    for file in os.listdir(folderPath):
        with open(os.path.join(folderPath, file), 'r') as f:

            text = f.read()
            poemList.append(text)
    return poemList

