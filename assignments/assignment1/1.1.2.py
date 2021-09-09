"""
1.1 Finding collocations (2 points)

• Create a tool to find collocations using hypothesis testing 
    (see slides for Lecture 2). Use the Brown corpus (already in NLTK).
• Consider sequences of 2 words (bigrams).
• Generate files containing the collocations.
"""
import math
import nltk
from nltk.corpus import brown
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()

words = brown.words()
finder = BigramCollocationFinder.from_words(words)
collocations = finder.nbest(bigram_measures.pmi, 100)

# Iterating over corpus many times is a bad way to do this.
def count_collocation(first, second):
    cnt = 0
    for i in range(len(words) - 1):
        if words[i] == first and words[i+1] == second:
            cnt += 1
    return cnt

# Iterating over corpus many times is a bad way to do this.
def test_collocation(collocation):
    first, second = collocation
    first_count = words.count(first)
    second_count = words.count(second)

    p_first = first_count / 1161192
    p_second = second_count / 1161192
    p_collocation = p_first * p_second
    pt_collocation = count_collocation(first, second) / 1161192
    t = (pt_collocation - p_collocation) / (math.sqrt(pt_collocation) * math.sqrt(1 / 1161192))
    return t > 2.576

correct_collocations = [collocation for collocation in collocations if test_collocation(collocation)]

f = open('1.1.2.txt', 'w')
for collocation in correct_collocations:
    f.write(str(collocation))
    f.write('\n')

f.close()