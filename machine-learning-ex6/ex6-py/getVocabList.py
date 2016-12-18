def getVocabList():
    with open('vocab.txt') as f:
        vocabList = []
        for line in f:
            idx, w = line.split()
            vocabList.append(w)
    return vocabList
