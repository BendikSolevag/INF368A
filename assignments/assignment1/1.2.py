

import nltk
from nltk.util import bigrams
from nltk.corpus import wordnet as wn


list_of_collocations = [['agreeable', 'autocracies'],
['estate', 'boards'],
['furious', 'overfall'],
['grands', 'crus'],
['klux', 'klan'],
['ku', 'klux'],
['longue', 'vue'],
['sancho', 'panza'],
['sultan', 'ahmet'],
['bel', 'canto'],
['andrea', 'palladio'],
['bam', 'ld'],
['enver', 'pasha'],
['polo', 'grounds'],
['sara', 'sullam'],
['woonsocket', 'patriot'],
['beech', 'pasture'],
['chemische', 'krystallographie'],
['duncan', 'phyfe'],
["groth's", 'chemische'],
['hwang', 'pah'],
['baton', 'rouge'],
['dolce', 'vita'],
['final', 'solution'],
['patterson', 'moos'],
['ol', 'slater'],
['coral', 'gables'],
['v-shaped', 'inlet'],
['crystal', 'structures'],
["kaiser's", 'fountain'],
['nineteen', 'eighty-four'],
['notre', 'dame'],
["salyer's", 'canyon'],
['loom', 'winder'],
['neutral', 'tones'],
['souvanna', 'phouma'],
['las', 'vegas'],
['sargent', 'shriver'],
['ham', 'richert'],
['sterling', 'township'],
['train', 'robbery'],
['american-negro', 'suite'],
['moise', 'tshombe'],
['rabbi', 'melzi'],
['real', 'estate'],
['strong', 'tea']]

"""
1 Assignment Description

1.2 Correction tool (2 points)

• Create a simple tool that corrects non-natural expressions. In detail, your tool should receive as input two or three words. 
    If there is a collocation in your files such that the i-th word is a synonym of the i-th word given as input then the algorithm 
    will output the first such collocation in your files (consider that two words that are the same are synonyms). For example, 
    if it receives “powerful tea” and “strong tea” is in your list then the algorithm should print “strong tea”.
• Suggestion: Use WordNet to detect synonyms.
"""

"""
Generate bigrams from input sentence. One at a time, generate synonyms for words in bigrams. 
Run new bigram, one word replaced with synonym against collocations.
If a match occurs, return the new bigram. Otherwise keep the old bigram. 

"""

input_sentence = 'Expensive real demesne'
print(input_sentence)
tokenized_input = nltk.word_tokenize(input_sentence)
for i in range(len(tokenized_input) - 1):
    # We test only on bigrams, as our reference collocations are all bigrams. 
    bigram = tokenized_input[i:i+2]
    # Iterate over all collocations we have
    for collocation in list_of_collocations:
        # Check if word matches word in collocation
        if(collocation[0] == bigram[0]):
            for synset in wn.synsets(bigram[1]):
                for lemma in synset.lemmas():
                    # Check if following word is synonym with following word in collocation
                    if(lemma.name() == collocation[1]):
                        #If true, replace input sentence with collocation
                        tokenized_input[i:i+2] = [bigram[0], collocation[1]]
        if(collocation[1] == bigram[1]):
            for synset in wn.synsets(bigram[0]):
                for lemma in synset.lemmas():
                    if(lemma.name() == collocation[0]):
                        tokenized_input[i:i+2] = [collocation[0], bigram[1]]

print(' '.join(tokenized_input))

"""
for synset in wn.synsets('demesne'):
    print(synset)
    for lemma in synset.lemmas():
        print(lemma.name())
"""