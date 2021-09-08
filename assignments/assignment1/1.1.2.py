"""
1.1 Finding collocations (2 points)

• Create a tool to find collocations using hypothesis testing 
    (see slides for Lecture 2). Use the Brown corpus (already in NLTK).
• Consider sequences of 2 words (bigrams).
• Generate files containing the collocations.
"""
import math
import nltk
from nltk.util import bigrams
from nltk.corpus import brown
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()


finder = BigramCollocationFinder.from_words(brown.words())
finder.apply_freq_filter(3)
collocations = finder.nbest(bigram_measures.pmi, 50)

# Iterating over corpus many times is a bad way to do this.
def test_collocation(collocation):
    first, second = collocation
    p_first = brown.words().count(first) / 1161192
    p_second = brown.words().count(second) / 1161192
    p_collocation = p_first * p_second
    pt_collocation = brown.words().count(first + ' ' + second) / 1161192
    t = (pt_collocation - p_collocation) / (math.sqrt(pt_collocation) * math.sqrt(1 / 1161192))
    return t > 2.576

correct_collocations = [collocation for collocation in collocations if test_collocation(collocation)]

f = open('1.1.2.txt', 'w')
for collocation in correct_collocations:
    f.write(str(collocation))
    f.write('\n')

f.close()