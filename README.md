# implementation-word2vec-cbow-skipgram-pytorch

## Contents
* About Project
* Motivation
* Algorithm
* Tech/framework used

&nbsp;

## About Project
Implemented word2vec cbow and skipgram using pytorch.

&nbsp;

## Motivation
School Project

&nbsp;

## Algorithm
Used text8 data to train. Implemented CBOW and Skipgram alogrithms.

&nbsp;

**How to use**
run word2vec code with python.
~~~
python word2vec.py
~~~
Then, below word2vec_trainer function will run. Can change parameters.

~~~
word2vec_trainer(corpus, word2ind, codes=None,freqtable = None, nonleaf_ind = None, mode="CBOW", mode2 = "None",use_subsample=None ,dimension=100, learning_rate=0.025, iteration=50000)

corpus = text8
mode = 'CBOW' or 'Skipgram'
learning_rate = (default) 0.025
iteration = (default) 50000
dimension = (default) 100
~~~

&nbsp;

## Tech/framework used
<b>Built with</b>
- python
- pytorch
- numpy
- huffman
&nbsp;
