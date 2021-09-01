
import nltk
from nltk.collocations import *

# Exercise 1
sentence = "In Dusseldorf I took my hat off. But I can’t put it back on."
tokens = nltk.word_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged[0:6])

# Exercise 2
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_meaures = nltk.collocations.TrigramAssocMeasures()
finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))
print(finder.nbest(bigram_measures.pmi, 10))

finder.apply_freq_filter(3)
print(finder.nbest(bigram_measures.pmi, 10))
