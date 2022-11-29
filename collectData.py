
def collectDataFromTextDoc():
    with open("Edgar Allan Poe complete works.txt") as f:
        text = f.read()
        f.close()
    return text

