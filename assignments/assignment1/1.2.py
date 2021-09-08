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
from nltk.corpus import brown

# Generate list of common collocations over dataset. Take as input a few words, look up the combinations of words in our list. 
# If any word appears, give as output the identified collocation.
