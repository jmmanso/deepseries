
# Text sequencing with DeepSeries

### In this example we show how to train DeepSeries on a text corpus and let the network predict future sentences. We use Alice in Wonderland and train on character series, so that in the end we can input a leading sentence and get the continuation from the network.



```python
import numpy as np
import os, sys
from deepseries import deepseries, preprocessing
import encoder
```


```python
# Load Alice in Wonderland
with open('alice_in_wonderland_FORMATTED.txt', 'rb') as f:
    text_corpus = f.read()
```


```python
# Feed into encoder class. The code for this is in a separate file.
# It performs transformations between text characters and one-hot binary
# encoded labels.
Enc = encoder.Encoder(text_corpus)
```


```python
# split string into sequences, like 15 characters
X,y = preprocessing.Slicer().fit(Enc.onehot_corpus, sequence_length=15,kind='seq2seq')
```


```python
# Test/train split. Test is the trailing 20% of the corpus
split_index = int(0.8*X.shape[0])
Xtrain, Xtest, ytrain, ytest =  \
X[:split_index], X[split_index:], y[:split_index], y[split_index:]
```


```python
# Initialize DeepSeries sequencer with classification learning.
# Increasing Nnodes or Nlayers will increase the size (and complexity)
# of the network
ds = deepseries.Sequencer(Xtrain, ytrain, Xtest, ytest,
    Nnodes=150,
    Nlayers=2,
    input_keep_prob=.9,
    output_keep_prob=.9,
    optimizer_type='AdamOptimizer',
    optimizer_kwargs={'learning_rate':0.001},
    cell_kwargs = {'use_peepholes':False,
                   'forget_bias':1,
                   'state_is_tuple':True,
                   'activation':deepseries.tf.nn.relu},
    batch_size=60,
    cell_type='GRUCell',
    tracking_step=200)
```


```python
# Do some light fitting. If you are running the code yourself,
# start with one epoch to test your computing speed.
ds.fit(n_epochs=2)
```

    train batch cost: 2944.771, test cost: 1330471.275
    train batch cost: 1985.973, test cost: 910561.092
    train batch cost: 1785.021, test cost: 819863.075
    train batch cost: 1702.215, test cost: 787322.093
    train batch cost: 1599.467, test cost: 759941.718
    train batch cost: 1593.731, test cost: 747247.351
    train batch cost: 1624.323, test cost: 734177.895
    train batch cost: 1539.692, test cost: 729728.033
    train batch cost: 1435.394, test cost: 717685.258
    train batch cost: 1560.239, test cost: 713081.357
    train batch cost: 1496.891, test cost: 712338.882
    train batch cost: 1473.880, test cost: 709090.291
    train batch cost: 1423.819, test cost: 704028.349
    train batch cost: 1538.557, test cost: 699864.659
    train batch cost: 1456.718, test cost: 696315.014
    train batch cost: 1414.644, test cost: 692815.532
    train batch cost: 1392.590, test cost: 689580.917
    train batch cost: 1412.664, test cost: 689562.559
    train batch cost: 1477.363, test cost: 687777.463
    train batch cost: 1447.342, test cost: 687588.822



```python
# Let's see what the network outputs if we feed in
# chunks from the test set:
test_sample = Xtest[::10][:10]
input_sentences = Enc.onehot2chars(test_sample)
output_sentences = Enc.onehot2chars(ds.unroll(test_sample, n_output=15))
zip(input_sentences, output_sentences)
```




    [('ot thrown out t', 'he said the moc'),
     ('out to sea so t', 'he same to hers'),
     (' so they had to', ' say would to h'),
     ('ad to fall a lo', 'ng the morse sa'),
     (' a long way so ', 'she was to hers'),
     ('y so they got t', 'he door and the'),
     ('got their tails', ' in t at the ro'),
     ('tails fast in t', 'he door and too'),
     (' in their mouth', ' and the rabbit'),
     ('mouths so they ', 'would the rabbi')]




```python
# We can also use this wrapper method to input
# any sentence you can think of, of any length.
# You can also set any number of output characters:
Enc.talk(ds, 'alice was drinking tea and ', 30)
```




    'the reased a little said the r'



### The network trained only on character sequence, but it was able to learn the structure of words and how to put them together, handling blank spaces quite well.
