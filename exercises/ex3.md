<h1>Exercise 3</h1>
<section>
    <h2>Ex. 3.1</h2>
    <p>
    • What is the probability of ‘i want english food’?
        0.25 * 0.33 * 0.0011 * 0.5 = 4.5375000000000005e-05
    • What is the probability of ‘i want chinese food’?
        0.25 * 0.33 * 0.0065 * 0.52 = 0.00027885000000000003
    • Discuss what is influencing the probabilities.
    </p>
</section>

<section>
    <h2>Ex. 3.2</h2>
    <p>
    Add-k: k is added to all bigram counts to ensure no bigrams has zero probability of occuring.
    Backoff: If there are no occurences of a n-gram, we use less context by lowering n, thereby 'backing off'
    Interpolation: Mix the probability estimates from the weighted combination of all n-grams.
    </p>
</section>



<section>
    <h2>Ex. 3.3</h2>
    <p>
    </p>
</section>


<section>
    <h2>Ex. 3.4</h2>
    Held out corpora is what we traditionally refer to as the test set in machine learning. It is a set of data the model has not been trained on, used to test its performance. Training on a test set is whenever sentences from the test set also occurs in the training set, which may artificially improve a model's performance.
    

    The perplexity of a language model on a test set is the inverse probability of the test set, normalized by the number of words. How well a language model can predict the next word and therefore make a meaningful sentence is asserted by the perplexity value assigned to the language model based on a test set.
    
    We should not consider the vocabulary of the test in the construction of a n-gram model, as this may artificially improve its performance.

    PP = (product(1/P(element)))^(1/10) = 0.1

</section>


