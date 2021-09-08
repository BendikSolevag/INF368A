


"""
1.1 Finding collocations (2 points)

• Create a tool to find collocations using (1) frequency plus part-of-speech tagging (search for adjectives and nouns) and (2) hypothesis testing (see slides for Lecture 2). Use the Brown corpus (already in NLTK).
• Consider sequences of 2 words (bigrams).
• Generate files containing the collocations.
"""

import nltk
from nltk.util import bigrams
from nltk.corpus import brown
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()

finder = BigramCollocationFinder.from_words(brown.words())
finder.apply_freq_filter(3)
collocations = finder.nbest(bigram_measures.pmi, 50)

tagged_collocations = [nltk.pos_tag(bigram) for bigram in collocations]

including = ['NN', 'NNS', 'JJ']
def check(bigram):
    for _, cl in bigram:
        result = False
        for cls in including:
            if cls in cl:
                result = True
        if not result:
            return result
    return True

correct_bigrams = [bigram for bigram in tagged_collocations if check(bigram)]

f = open('1.1.1.txt', 'w')
for bigram in correct_bigrams:
    f.write(str(bigram))
    f.write('\n')

f.close()
