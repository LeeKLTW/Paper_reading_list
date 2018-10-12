Sutskever et al,Sequence to sequence learning with neural networks


2 The model
A simple strategy for general sequence learning is to map the input sequence to a fixed-sized vector
using one RNN, and then to map the vector to the target sequence with another RNN (this approach
has also been taken by Cho et al. [5] While it could work in principle since the RNN is provided
with all the relevant information, it would be difficult to train the RNNs due to the resulting long
term dependencies [14, 4] (figure 1) [16, 15]. However, the Long Short-Term Memory (LSTM) [16]
is known to learn problems with long range temporal dependencies, so an LSTM may succeed in
this setting.
...

Our actual models differ from the above description in three important ways. First, we used two
different LSTMs: one for the input sequence and another for the output sequence, because doing
so increases the number model parameters at negligible computational cost and makes it natural to
train the LSTM on multiple language pairs simultaneously [18]. Second, we found that deep LSTMs
significantly outperformed shallow LSTMs, so we chose an LSTM with four layer

...

3.3 Reversing the Source Sentences
While the LSTM is capable of solving problems with long term dependencies, we discovered that
the LSTM learns much better when the source sentences are reversed (the target sentences are not
reversed). By doing so, the LSTM’s test perplexity dropped from 5.8 to 4.7, and the test BLEU
scores of its decoded translations increased from 25.9 to 30.6.

...
While the LSTM is capable of solving problems with long term dependencies, we discovered that
the LSTM learns much better when the source sentences are reversed (the target sentences are not
reversed). By doing so, the LSTM’s test perplexity dropped from 5.8 to 4.7, and the test BLEU
scores of its decoded translations increased from 25.9 to 30.6.
While we do not have a complete explanation to this phenomenon, we believe that it is caused by
the introduction of many short term dependencies to the dataset. Normally, when we concatenate a
source sentence with a target sentence, each word in the source sentence is far from its corresponding
word in the target sentence. As a result, the problem has a large “minimal time lag” [17]. By
reversing the words in the source sentence, the average distance between corresponding words in
the source and target language is unchanged. However, the first few words in the source language
are now very close to the first few words in the target language, so the problem’s minimal time lag is
greatly reduced
...
