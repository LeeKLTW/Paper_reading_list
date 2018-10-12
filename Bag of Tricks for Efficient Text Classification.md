2 Model architecture

This architecture is similar to
the cbow model of Mikolov et al. (2013), where
the middle word is replaced by a label.

2.1 Hierarchical softmax
...
In
order to improve our running time, we use a hierarchical
softmax (Goodman, 2001) based on the
Huffman coding tree (Mikolov et al., 2013).

...
Note:
O(kh) -> O(h log2(k))

k:# of classes
h:hidden nodes


2.2 N-gram features
Bag of words is invariant to word order but taking
explicitly this order into account is often computationally
very expensive. Instead, we use a
bag of n-grams as additional features to capture
some partial information about the local word order.

...

Note:
Table 1:accuracy, same or better
Table 2:Trainig time, extremely less!
