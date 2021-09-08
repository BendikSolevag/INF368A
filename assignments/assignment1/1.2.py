"""
1 Assignment Description

1.2 Correction tool (2 points)

• Create a simple tool that corrects non-natural expressions. In detail, your tool should receive as input two or three words. 
    If there is a collocation in your files such that the i-th word is a synonym of the i-th word given as input then the algorithm 
    will output the first such collocation in your files (consider that two words that are the same are synonyms). For example, 
    if it receives “powerful tea” and “strong tea” is in your list then the algorithm should print “strong tea”.
• Suggestion: Use WordNet to detect synonyms.

"""

import nltk
from nltk.util import bigrams
from nltk.corpus import wordnet as wn


"""
Generate bigrams from input sentence. One at a time, generate synonyms for words in bigrams. Run new bigram, one word replaced with synonym against collocations.
If a match occurs, return the new bigram. Otherwise keep the old bigram. 

"""


print(wn.synsets('powerful'))