---
layout: post
title: Hacker's Guide to (Almost) all Neural Nets
author: Jeffrey R. Norton
---
The purpose of this document is to do a general review of as all neural nets.  This is a difficult prospect because the field moves quickly.  The objective is to provide guidance in designing neural networks.  The first step is to list the types and break down the functionality and objectives of the various parts of the neural network.  And as possible, there will be code fragments or references to frameworks such as Keras or Tensorflow for example solutions.

![](http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png)

**References**

**Fjodor Vanveen. Neural Network Prequel: Cells and Layers. The Asimov Institute. March 31, 2017. [https://www.asimovinstitute.org/author/fjodorvanveen/](https://www.asimovinstitute.org/author/fjodorvanveen/).*

*Andrew Tchircoff. The mostly complete chart of Neural Networks, explained. Towards Data Science. Aug 4, 2017. [https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464).*

*Wikipedia. Types of artificial neural networks. Accessed June 2018. [https://en.m.wikipedia.org/wiki/Types_of_artificial_neural_networks](https://en.m.wikipedia.org/wiki/Types_of_artificial_neural_networks).*

## Order of Presentation

The networks are presented from a pseudo-historical point of view.  The reason is to try to frame the philosophy of statistical data computing over the years.  But we say pseudo-historical as to some extent we will group similar technologies.

## Robust Code in Python

Let's start with good practices in writing robust code for machine learning, even in a notebook.  In general these principles can be used in many languages, but we demonstrate the principles in Python 3.

### Use Functions
Notebooks tell a story and it is easy just to weave that story in a chronological stream of thought and development.  However, such development has some drawbacks.  First, it will lead to repetition of code as you experiment with different hyperparameters in your coding.  This is of course a bad practice in software development.

A better practice in the author's opinion is to adopt the following principles in writing code, principles which data engineers will recognize as the foundations to writing microservices.  Adopting this approach allows those same data engineers to directly adapt code written by the data scientist into commercial code.  These practices will actually make it easier to maintain the code even in a notebook.

#### Code Decomposition
Break down the code into *bounded concerns*.  Simply, this is the decomposition of the logic into small standalone pieces.  The actual objective is to remove dependencies (as much as possible) between the different pieces of the solution.  Microservice solutions further separate these pieces by using interfaces such as ReST or GraphQL.  Indeed, if we could write our notebooks with these principles in mind, it will lead to far better code.

To some extent, we can practice this principle by storing intermediate steps to disk.  For example, consider a deep learning problem where we 

1. Read data from a foreign source and *store the data to disk in a standard format*.
2. Read the just-stored data and perform data cleaning and imputation as required.  *Store that modified data to disk in a standard format.*
3. Create the desired model or models and store them to disk in the framework format.
4. Load the model and train it.  Store the resulting model definition (weights) to disk.
5. Run the test data and measure model performance.  Store those results to disk.

It can immediately be seen that adopting this practice allows us to execute different pieces of the solution in different sessions.  It allows us to consider different models.  Furthermore, for pieces which are not critical parts of the solution, we can write them as Python scripts to be executed outside the notebook, hiding the less-interesting parts of the code.

#### More Efficient Ways to Store Data

With Python, one potential way to store data is as Pickle files.  Frameworks such as Keras have methods to store models, but using Pickle to store certain data constructs is a viable and simple method.

In 2018, it appears that Parquet is the new rave.  Part of the reason is because Parquet integrates very well with Apache Arrow which is optimized for delivering performance when working with large datasets.  However, for this relatively small dataset, we don't get any performance enhancements.

These functions are ideal to write as Python scripts as it is most likely that in a notebook, how the data is accessed is not as important as the topology of the network.  The key is that for datasets which are to be accessed from the web, performance-wise it is better to bring the file local and using Parquet allows us to compress and still use the data efficiently.

**References**

Wikipedia, Representational state transfer. Accessed July 2018.  [https://en.wikipedia.org/wiki/Representational_state_transfer](https://en.wikipedia.org/wiki/Representational_state_transfer).

Wikipedia, GraphQL. Accessed July 2018. [https://en.wikipedia.org/wiki/GraphQL](https://en.wikipedia.org/wiki/GraphQL).

Python Documentation. 12.1 pickle - Python object serialization. Accessed July 2018. [https://docs.python.org/3/library/pickle.html#data-stream-format](https://docs.python.org/3/library/pickle.html#data-stream-format)

Arrow Documentation. Reading and Writing the Apache Parquet Format. Accessed July 2018. [https://arrow.apache.org/docs/python/parquet.html](https://arrow.apache.org/docs/python/parquet.html).

Apache Arrow. Accessed July 2018. [https://arrow.apache.org/](https://arrow.apache.org/).

#### Assertions and Exception Handling

Python is a scripting language and as such, lacks some of the nice features of compiled languages which means that we must rely on assertions not only to make sure that data exists and is complete, but also to make sure it is of the correct type.

Exception handling should always be employed and as much as possible, it needs to be very tight and specific.  By tight, I mean that smaller logic blocks should be in their own try block.  By specific, the exceptions should be of a certain type, e.g., ```TypeError``` rather than just using the general ```Exception``` class.

#### Logging

In production code, logging is critical.  Logging is a wonderful tool for not only for diagnosing failures, but is also wonderful for debugging code.

The following code fragment shows how to set up logging in a Jupyter notebook.  For our purposes, we will use a very straightforward approach using a yaml configuration.  All utilities and configuration files are in the folder ```utilities```.

With all this in mind, we can write a robust utility for working with Pickle and Parquet files.  Note that *if the files are too large*, then it will be necessary to process row subsets if not on a large cluster.  See [Reading and Writing the Apache Parquet Format](https://arrow.apache.org/docs/python/parquet.html) for details on how to process row subsets.

> **Code Fragment 1.1** - ```utilities/pickle_util.py```

```python
# Import the Pickle library
import pickle                                                                           import pandas as pd                                                                     import time

""" @description                                                                                                                             
     read_csv_to_pickle
     Read a data file from the UCI datasets and store it into a pickle file.
    @parameter url - The URL of the CSV
    @parameter names - Column names
    @parameter store_path - The local path for the Pickle file
    @parameter timeit - True if timing operations
"""
def read_csv_into_pickle(url, names, store_path, timeit=True):
    try:
        if timeit: start=time.time()
        df = pd.read_csv(url, names=names)
        if timeit: print('URL Read Time = {} seconds'.format(time.time()-start))
        if timeit: start = time.time()
        with open(store_path, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if timeit: print('Pickle Creation Time = {} seconds'.format(time.time()-start))
    except TypeError as t_exception:
        logging.error(str(t_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))

""" @description
     read_pickle_to_df
     Read data from a local pickle file into a data frame
    @parameter store_path - The local path for the Pickle file
    @parameter timeit - True if timing operations
"""
def read_pickle_to_df(store_path, timeit=True):
    try:
        # Read the data from the Pickle file
        start = time.time()
        with open(store_path, 'rb') as handle:
            df = pickle.load(handle)
        print('Pickle Read Time = {} seconds'.format(time.time()-start))
        return df
    except FileNotFoundError as fnf_exception:
        logging.error(str(fnf_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))
    return None
```

> **Code Fragment 1.2** - ```utilities/parquet_util.py```

```python
import pandas as pd
import time
import logging

""" @description
     read_csv_to_parquet
     Read a data file from the UCI datasets and store it into a parquet file.
    @parameter url - The URL of the CSV
    @parameter names - Column names
    @parameter store_path - The local path for the Parquet file
    @parameter timeit - True if timing operations
"""
def read_csv_into_parquet(url, names, store_path, timeit=True):
    try:
        if timeit: start=time.time()
        df = pd.read_csv(url, names=names)
        if timeit: logging.info('URL Read Time = {} seconds'.format(time.time()-start))
    except FileNotFoundError as fnf_exception:
        logging.error(str(fnf_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))
    try:
        if timeit: start = time.time()
        df.to_parquet(store_path, compression='gzip')
        if timeit: logging.info('Parquet Creation Time = {} seconds'.format(time.time()-start))
    except TypeError as t_exception:
        logging.error(str(t_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))

""" @description
     read_parquet_to_df
     Read data from a local pickle file into a data frame
    @parameter store_path - The local path for the Parquet file
    @parameter timeit - True if timing operations
"""
def read_parquet_to_df(store_path, timeit=True):
    try:
        # Read the data from the Pickle file
        start = time.time()
        df = pd.read_parquet(store_path)
        if timeit: logging.info('Parquet Read Time = {} seconds'.format(time.time()-start))
        return df
    except FileNotFoundError as fnf_exception:
        logging.error(str(fnf_exception))
    except Exception as unclassified_exception:
        logging.error('Unclassified exception\n{}'.format(str(unclassified_exception)))
    return None
```

> **Code Fragment 1.3** - ```utilities/logging.yaml``` is the configuration file for logging.

```json
version: 1
disable_existing_loggers: False
formatters:
    simple:
        format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

handlers:
    console:
        class: logging.StreamHandler
        level: DEBUG
        formatter: simple
        stream: ext://sys.stdout

    info_file_handler:
        class: logging.handlers.RotatingFileHandler
        level: INFO
        formatter: simple
        filename: info.log
        maxBytes: 10485760 # 10MB
        backupCount: 20
        encoding: utf8

    error_file_handler:
        class: logging.handlers.RotatingFileHandler
        level: ERROR
        formatter: simple
        filename: errors.log
        maxBytes: 10485760 # 10MB
        backupCount: 20
        encoding: utf8

loggers:
    my_module:
        level: ERROR
        handlers: [console]
        propagate: no

root:
    level: INFO
    handlers: [console]
```
> **Code Fragment 1.4** - ```utilities/setup_logging.py``` uses the yaml file to configure logging and can be used in the notebook.

```python
import os
import logging.config

import yaml

def setup_logging(
    default_path='logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    logger = logging.getLogger(__name__)
```

For Python scripts, add an ```__init__.py``` file to the utilities folder.  Add this code as ```logging.py``` in the utilities folder.  Add all utilities now into the same folder.

> **Code Fragment 1.5** - Robust Jupyter notebook to read UCI iris data into a local Parquet file.

Make sure that pip is upgraded, that we have the latest version of pandas and that pyarrow is installed.  pyarrow is required in order to create parquet files directly from pandas as is done in the utility.

```python
!pip install --upgrade pip
```

```
Requirement already up-to-date: pip in /opt/conda/lib/python3.5/site-packages (18.0)
```



```python
!pip install pandas --upgrade
```

```
Requirement already up-to-date: pandas in /opt/conda/lib/python3.5/site-packages (0.23.3)
Requirement already satisfied, skipping upgrade: pytz>=2011k in /opt/conda/lib/python3.5/site-packages (from pandas) (2016.10)
Requirement already satisfied, skipping upgrade: numpy>=1.9.0 in /opt/conda/lib/python3.5/site-packages (from pandas) (1.11.2)
Requirement already satisfied, skipping upgrade: python-dateutil>=2.5.0 in /opt/conda/lib/python3.5/site-packages (from pandas) (2.6.0)
Requirement already satisfied, skipping upgrade: six>=1.5 in /opt/conda/lib/python3.5/site-packages (from python-dateutil>=2.5.0->pandas) (1.10.0)
```



```python
!pip install pyarrow --upgrade
```

```
Requirement already up-to-date: pyarrow in /opt/conda/lib/python3.5/site-packages (0.9.0)
Requirement already satisfied, skipping upgrade: numpy>=1.10 in /opt/conda/lib/python3.5/site-packages (from pyarrow) (1.11.2)
Requirement already satisfied, skipping upgrade: six>=1.0.0 in /opt/conda/lib/python3.5/site-packages (from pyarrow) (1.10.0)
```

Set up logging

```python
import logging
import utilities.setup_logging

utilities.setup_logging.setup_logging()
```

Use our script utilities to read the UCI iris database into Pickle and Parquet files.

```python
from utilities.pickle_util import read_csv_into_pickle

pickle_path = 'iris.pickle'

# Read the CSV (at URL) into a Pickle file
read_csv_into_pickle(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
    store_path=pickle_path
)
```

```
INFO:root:URL Read Time = 0.1403353214263916 seconds
INFO:root:Pickle Creation Time = 0.0016875267028808594 seconds
```



```python
!ls -l iris.pickle
```

```
-rw-r--r-- 1 root root 5997 Jul 23 15:20 iris.pickle
```



```python
from utilities.parquet_util import read_csv_into_parquet

parquet_path = 'iris.parquet'

# Read the CSV (at URL) into a Parquet file
read_csv_into_parquet(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
    store_path=parquet_path
)
```

```
INFO:root:URL Read Time = 0.12006831169128418 seconds
INFO:root:Parquet Creation Time = 0.022646427154541016 seconds
```



```python
!ls -l iris.parquet
```

```
-rw-r--r-- 1 root root 4144 Jul 23 15:20 iris.parquet
```

Read the Pickle and the Parquet files into dataframes.

```python
from utilities.pickle_util import read_pickle_to_df

pickle_df = read_pickle_to_df(pickle_path)
```

```
INFO:root:Pickle Read Time = 0.0009696483612060547 seconds
```



```python
from IPython.display import display

display(pickle_df.head())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



```python
from utilities.parquet_util import read_parquet_to_df

parquet_df = read_parquet_to_df(parquet_path)
```

```
INFO:root:Parquet Read Time = 0.006642341613769531 seconds
```



```python
display(parquet_df.head())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>

Use assertions to make sure that all the data is as expected.

```python
import pandas as pd

assert(isinstance(pickle_df, pd.DataFrame))

assert(isinstance(parquet_df, pd.DataFrame))

assert(pickle_df.equals(parquet_df))
```
