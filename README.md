# DeepSeries

This python package is built on top of Tensorflow and implements Long-short-term-memory neural networks for time series prediction. The learners can be found in deepseries.py, while some data pre-processing operations are implemented in preprocessing.py.

Cool features include:
* Ability to take both continuous and categorical data (or a mixture) as input and output.
* Option to feed arbitrary weights for the data samples at the training stage.
* Although training data needs to be input as sequences of fixed size, the prediction methods can generate output sequences of arbitrary length.

Please take a look at the *examples/* folder for simple demos, which include text sequencing, time series classification and time series sequencing.

### Installation
Install TensorFlow. Then, clone or download this repo on your system. Go inside the root directory of the package, and type
```
pip install .
```

### Main classes:
#### DeepSeries
Base class supporting the core functionality. Not meant to be used directly.

#### Sequencer
Takes a series as input, and continues to generate new points into the "future". For training, this class needs to be fed with sequence chunks of a fixed size (the class preprocessing.Slicer can be used for that). The length of each chunk determines the extent to which the network will try to keep a memory of the sequence being analyzed.
For predictions, we need to supply the initial sequence that we want to extend forward. Unlike during training, you may use any length for the initial sequence, as well as for the output. However, best results are achieved if the input sequence has the same length as the training chunk size. The motivation to implement this flexibility is that, in real-world time series problems, not all the series points are always available when we need to predict a sample. *deepseries.Sequencer* offers the possibility of predicting with whatever information is ready.
Sequences can be either real numbers, one-hot encoded classes, or a mixture of these.

#### many2oneClassifier
This classifier tries to predict a label based on a sequence of points. While *deepseries.Sequencer* is a sequence-to-sequence predictor (where the sequence is the same), *deepseries.many2oneClassifier* does sequence-to-label. Here, the sequence can be composed of multiple time series channels, with real number data or categorical (one-hot encoded). The target labels must be one-hot encoded. This class will deliver label predictions, as well as class-probabilities.

#### many2oneRegressor
This is similar to *many2oneClassifier*, but instead of labels, it targets real numbers, as in sequence-to-number. It can also take multiple time series channels as input.
