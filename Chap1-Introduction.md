---
layout: post
title: Hacker's Guide to (Almost) all Neural Nets
author: Jeffrey R. Norton
---
The purpose of this document is to do a general review of as all neural nets.  This is a difficult prospect because the field moves quickly.  The objective is to provide guidance in designing neural networks.  The first step is to list the types and break down the functionality and objectives of the various parts of the neural network.  And as possible, there will be code fragments or references to frameworks such as Keras or Tensorflow for example solutions.

![](images/neuralnetworks.png)

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
<<../src/utilities/pickle_util.py>>
```

> **Code Fragment 1.2** - ```utilities/parquet_util.py```

```python
<<../src/utilities/parquet_util.py>>
```

> **Code Fragment 1.3** - ```utilities/logging.yaml``` is the configuration file for logging.

```json
<<../src/utilities/logging.yaml>>
```

> **Code Fragment 1.4** - ```utilities/setup_logging.py``` uses the yaml file to configure logging and can be used in the notebook.

```python
<<../src/utilities/setup_logging.py>>
```

For Python scripts, add an ```__init__.py``` file to the utilities folder.  Add this code as ```logging.py``` in the utilities folder.  Add all utilities now into the same folder.

> **Code Fragment 1.5** - Robust Jupyter notebook to read UCI iris data into a local Parquet file.

<<WRK/Chapter1.md>>

