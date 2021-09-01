<h1>Exercises 2</h1>
<section>
    <h2>Exercise 2.1</h2>
    <p>
        A colocation is an expression consisting of two or more words that correspond to some conventional way of saying things. (Firth 1957)
        Collocations are characterised by limited compositionality. A language expression is compositional if the meaning of the expression can be predicted from the meaning of the parts.
        White wine is a collocation, as the word "white" cannot be exchanged while keeping the phrase`s meaning.
        A (adjective) N (noun)
        N N
        A A N
        A N N
        N A N
        N N N
        N P (preposiiton) N
    </p>
</section>
<section>
    <h2>Exercise 2.2</h2>
    <p>
        Analysing mean and variance gives an estimate of the average 'distance' between two words. We do this by calculating the mean distance between the words, as well as their standard deviation. We accept that high frequency and low variance could be accidental, so we test our pairs via hypothesis testing. We assume our co-occurrences are by chance, and compute the chance that the event would occur if our assumption were true. We reject the pair if the probability is too low. 
    </p>
</section>
<section>
    <h2>Exercise 2.3</h2>
    <p>
        'Expensive' occurs 14500 times.
        'Wine' occurs 4500 times.
        Corpus has 15000000 tokens.
        'Expensive Wine' occurs 7 times.

        Pr(expensive) = 14500/15000000 = 0.0009666666666666667
        Pr(wine) = 4500/15000000 = 0.0003
        P(expensive wine) = P(expensive) * P(wine) = 2.9 * 10^-7

        Pr(expensive wine) = 7/15000000 = 4.667 * 10^-7
        t = ((4.667 - 2.9) * 10^-7) / sqrt( (4.667 * 10^-7)/(15000000)) = 1.0017590592405523
        T value is below 2.576 -> 'Expensive wine' is not a collocation.
    </p>
</section>