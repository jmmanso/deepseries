
# Simple classification of time series with DeepSeries

### In this example train a classifier with a public data set of labeled numerical time series.
### A description of the data and labels can be found here: http://kdd.ics.uci.edu/databases/synthetic_control/synthetic_control.data.html
### It consists of 600 sequences with 60 time steps each. There are 6 labeled kinds of sequences.



```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os, sys
from deepseries import deepseries, preprocessing
```


```python
# Load data and scale each series so that its min/max are [-0.5,0.5]
df = pd.read_pickle('ts_data.pkl')
df = df - df.min()
df = df/df.max() - 0.5
```


```python
# Build an array of one-hot encoded labels
labels = df.index.values
enc = OneHotEncoder()
enc.fit([[l] for l in set(labels)])
onehot_labels = enc.transform(labels.reshape(-1,1)).toarray()
```


```python
# Reshape labels (y) and data (X) to
# (Nsequences=600, sequence_length=60, Nchannels=1) for X
# (Nsequences=600, sequence_length=1, Nchannels=6) for y
y = onehot_labels.reshape(600,1,6)
X = df.values.reshape(600,60,1)
```


```python
# Train-test split. Test data are the trailing 20% of samples
split_idx = int(0.8*X.shape[0])
Xtrain, Xtest, ytrain, ytest =  X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
```


```python
# Initialize classifier
dsc = deepseries.many2oneClassifier(
    Xtrain,
    ytrain,
    Xtest,
    ytest,
    Nnodes=50,
    Nlayers=2,
    optimizer_type='AdamOptimizer',
    optimizer_kwargs={'learning_rate':0.001},
    batch_size=40,
    tracking_step=20)
```


```python
# Fit classifier
dsc.fit(n_epochs=20)
```

    train batch cost: 67.970, test cost: 202.590
    train batch cost: 42.592, test cost: 114.336
    train batch cost: 28.770, test cost: 70.614
    train batch cost: 18.250, test cost: 48.772
    train batch cost: 15.075, test cost: 46.274
    train batch cost: 14.529, test cost: 31.100
    train batch cost: 5.646, test cost: 37.048
    train batch cost: 7.171, test cost: 33.351
    train batch cost: 6.589, test cost: 26.185
    train batch cost: 11.323, test cost: 35.252
    train batch cost: 5.731, test cost: 24.785
    train batch cost: 7.095, test cost: 25.040
    train batch cost: 3.395, test cost: 26.016
    train batch cost: 5.241, test cost: 23.565
    train batch cost: 9.701, test cost: 27.982
    train batch cost: 6.283, test cost: 29.715
    train batch cost: 5.597, test cost: 24.245
    train batch cost: 4.041, test cost: 22.840
    train batch cost: 3.276, test cost: 18.048
    train batch cost: 1.283, test cost: 21.016



```python
# Compute accuracy
dsc.classification_accuracy()
```




    0.94999999



### The classifier achieves around 95% accuracy on this 6-label classification. This performance is good considering the small size of the data set.


```python
# To return predictions in one-hot encoded form:
predictions_onehot = dsc.predict(Xtest)
print predictions_onehot[:3]
```

    [[[0 0 0 0 1 0]]

     [[0 0 0 0 1 0]]

     [[1 0 0 0 0 0]]]



```python
# To return predicted labels as sequential integers (e.g., the way they were before one-hot enconding in our case):
predictions = dsc.predict(Xtest, return_class_indices=True)
print predictions[:3]
```

    [[4]
     [4]
     [0]]



```python
# You also might want to output the softmax probabilities.
# These are the probability distributions, P(Class_i|Sample_k), and are
# normalized to 1 for every sample:
probabilities = dsc.predict_proba(Xtest)
print probabilities[:3]
```

    [[[  8.18560249e-04   4.89840677e-05   1.11215631e-03   6.32073011e-07
         9.98018861e-01   8.06433150e-07]]

     [[  8.60203363e-05   1.69548930e-05   8.69577343e-05   6.57956676e-08
         9.99810040e-01   2.27038335e-08]]

     [[  9.92291152e-01   4.66434238e-03   1.82475094e-04   4.42926335e-04
         2.28498410e-03   1.34180547e-04]]]
