""" The purpose of this file is to format the data for the DeepSeries example.

Data is sourced from http://kdd.ics.uci.edu/databases/synthetic_control/synthetic_control.html
and is delimited by spaces, but without a standard number of these. Thus, it cannot be 
easily read by pandas.DataFrame.read_csv. Here, we perform manual parsing of this file.
"""


import numpy as np
import pandas as pd


ts_data = []
with open('synthetic_control.data', 'r') as f:
    for line in f:
        # remove \n's and split around blank spaces
        tokens = line.strip('\n').split()
        # convert to floats
        numbers = [float(token) for token in tokens]
        ts_data.append(numbers)

ts_data = np.array(ts_data)

# According to the description in the website, the data
# has 600 records, where each chunk of 100 is linked to a
# particular label.
# Here we construct the 1-D array of labels
labels = np.zeros(len(ts_data), dtype=np.int32)
labels[0:100] = 0
labels[100:200] = 1
labels[200:300] = 2
labels[300:400] = 3
labels[400:500] = 4
labels[500:600] = 5

# Shuffle labels and data randomly
random_indices = np.arange(labels.size)
np.random.shuffle(random_indices)
labels = labels[random_indices]
ts_data = ts_data[random_indices]

# Save to a DataFrame, indexed by the labels
df = pd.DataFrame(ts_data, index=labels)
df.to_pickle('ts_data.pkl')
