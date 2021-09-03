"""
1 Assignment Description

1.2 Correction tool (2 points)

• Create a simple tool that corrects non-natural expressions. In detail, your tool should receive as input two or three words. If there is a collocation in your files such that the i-th word is a synonym of the i-th word given as input then the algorithm will output the first such collocation in your files (consider that two words that are the same are synonyms). For example, if it receives “powerful tea” and “strong tea” is in your list then the algorithm should print “strong tea”.
• Suggestion: Use WordNet to detect synonyms.

1.3 Report and lecturer’s evaluation (2 points)

• Write a report describing both the tool (algorithms, design choices) and how to use it. In more detail, your report should have: description of the problem; a description of your approach; a description of the software: installation, requirements, and usage notes; empirical evaluation. Your software should have the application (code and executable), the required libraries/software, and brief installation notes.

2 Evaluation schema
The evaluation of the results of your work will be evaluated by the lecturer and by your peers. The peer-review will not be used for grading your assignment but for providing you additional feedback. The evaluation schema is described as follows:


2.1 Lecturer’s Evaluation
• Presentation The quality and effectiveness of the presentation in describing the main features and characteristics of the assignment. Highlighting pro and cons, as well as outlining possible future extensions.
• Technical quality This include the organisation of the code in terms of readability
and commenting, as well as the appropriate use of external libraries and tools.
• Documentation Overall documentation of the project, including user manual (usage and installation) as well as the description of the main algorithms and techniques. Reference to relevant material (papers, web sites, libraries)


2.2 Peer evaluation (1 point)
You will get 1 point for writing the peer-reviews for your colleagues.
• Quality of documentation Whether the documentation explains the installation and the use of the software.
• Ease of use Is the software easy to use and to integrate into your programming environment and/or package you’re developing. The evaluation also includes the installation process and its pre-requisites (libraries and/or specific environments)
• Robustness How reliable is the software w.r.t. its specifications (i.e. “it does what it says on the tin”). How good it is at handling exceptions (e.g. error reporting).
• Effectiveness Whether the software is suitable for the tasks it is designed for and how effectively enables the user/other components to perform these tasks.
"""


"""
1.1 Finding collocations (2 points)

• Create a tool to find collocations using (1) frequency plus part-of-speech tagging (search for adjectives and nouns) and (2) hypothesis testing (see slides for Lecture 2). Use the Brown corpus (already in NLTK).
• Consider sequences of 2 words (bigrams).
• Generate files containing the collocations.
"""

import nltk
from nltk.util import bigrams, trigrams, ngrams
from nltk.corpus import brown

bigrams = list(bigrams(brown.words()))
tagged_bigrams = [nltk.pos_tag(bigram) for bigram in bigrams]

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

correct_bigrams = [bigram for bigram in tagged_bigrams if check(bigram)]

bigram_frequency = {}
for bigram in correct_bigrams:
    if(check(bigram)):
        try:
            bigram_frequency[str(bigram)] += 1
        except:
            bigram_frequency[str(bigram)] = 1
frequent_bigrams = sorted(bigram_frequency, key=bigram_frequency.get, reverse=True)[:50]

f = open('1.1.txt', 'w')
for bigram in frequent_bigrams:
    f.write(bigram)
    f.write('\n')

f.close()