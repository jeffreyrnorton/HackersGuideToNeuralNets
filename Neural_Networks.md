---
layout: post
title: A Review of (Almost) all Neural Nets
---
The purpose of this document is to do a general review of as many neural nets as possible.  The objective is to provide guidance in designing neural networks if possible.  The first step is to list the types and break down the functionality and objectives of the various parts of the neural network.  And as possible, there will be code fragments or references to frameworks such as Keras or Tensorflow for example solutions.

![](http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png)

#### References

*Fjodor Vanveen. Neural Network Prequel: Cells and Layers. The Asimov Institute. March 31, 2017. [https://www.asimovinstitute.org/author/fjodorvanveen/](https://www.asimovinstitute.org/author/fjodorvanveen/).*

*Andrew Tchircoff. The mostly complete chart of Neural Networks, explained. Towards Data Science. Aug 4, 2017. [https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464).*

*Wikipedia. Types of artificial neural networks. Accessed June 2018. [https://en.m.wikipedia.org/wiki/Types_of_artificial_neural_networks](https://en.m.wikipedia.org/wiki/Types_of_artificial_neural_networks).*

# Order of Presentation

The networks are presented from a pseudo-historical point of view.  The reason is to try to frame the philosophy of statistical data computing over the years.  But we say pseudo-historical as to some extent we will group similar technologies.

A probabilistic graphical models which is a probabilistic model for which a graph expresses the conditional dependence structure between random variables.  In general, there are two classes of models:  Markov fields and Bayesian networks.

### References

*Stanford. 1. Introduction. Accessed June 2018. http://pgm.stanford.edu/intro.pdf.*

*Daphne Koller, Nir Friedman, Lise Getoor, Ben Taskar. 2 Graphical Models in a Nutshell. Stanford. Accessed June 2018. https://ai.stanford.edu/~koller/Papers/Koller+al:SRL07.pdf.*

*Daphne Koller, Nir Friedman. Probabilistic Graphical Models: Principles and Techniques. The MIT Press. 2009. [https://www.amazon.com/Probabilistic-Graphical-Models-Principles-Computation/dp/0262013193](https://www.amazon.com/Probabilistic-Graphical-Models-Principles-Computation/dp/0262013193).*

# Markov Chain (MC)

<img src="https://cdn-images-1.medium.com/max/750/1*z5b8nmrykjNuU9y16-RVhQ.png" width="400px">

**Markov Property**

A stochastic process has the Markov property if the conditional probability distribution of the next state of the process depends only upon the present state, not on the sequence of events that preceded it. A process with this property is called a *Markov process*.  This is also referred to as a memoryless date.

>A **conditional probability distribution** is a probability distribution for a sub-population.  That is, a conditional probability distribution describes the probability that a randomly selected person from a *sub-population* has the *one characteristic of interest*. 

A [Markov network or chain](https://en.wikipedia.org/wiki/Markov_chain) may be represented by a graph that is undirected and may contain cycles.  This differs from Bayesian networks which we review shortly which is directed and acyclic (a DAG or directed-acyclic graph). Because of the different representations, a Markov network can represent (such as cyclic dependencies (which a Bayesian network cannot); on the other hand, a Markov chain cannot represent certain induced dependencies. The underlying graph of a Markov random field may be finite or infinite. 

A legacy application of Markov Chains are graphs where each edge has a probability. For example, a legacy application constructed texts like “after word *hello* we might have word *dear* with 0.0053% probability and word *you* with 0.03551% probability”.

Markov chains are not neural networks in a classic way, MCs can be used for classification based on probabilities (like [Bayesian filters](https://en.wikipedia.org/wiki/Bayes%27_theorem)), for clustering (of some sort), and as a finite state machine.

Markov chains are not considered to be neural networks, but they do resemble neural networks and form the theoretical basis for Hopfield Networks and Boltzmann networks which came later. Markov chains aren’t always fully connected either.

### References

*Hayes, Brian. “First links in the Markov chain.” American Scientist 101.2 (2013): 252.
[Original Paper PDF](http://www.americanscientist.org/libraries/documents/201321152149545-2013-03Hayes.pdf).*

*Phillipp von HIlgers and Amy N. Langville. The Five Greatest Applications of Markov Chains. Accessed June 2018. [http://langvillea.people.cofc.edu/MCapps7.pdf](http://langvillea.people.cofc.edu/MCapps7.pdf).*


# Hopfield Network (HN)

<img src="http://www.asimovinstitute.org/wp-content/uploads/2016/09/hn.png" width="400px">

A **Hopfield network (HN)** is a network where every neuron is connected to every other neuron as shown above - there are no layers.

A Hopfield network is trained by lowering the energy of states that the network should "remember".  The network will converted to a "remembered" state even if given only part of the state.  This makes the network a [content addressable memory system](https://en.wikipedia.org/wiki/Content-addressable_memory).

It is called *associative memory* because it recovers a state on the basis of similarity. For example, consider a Hopfield net with five units where state (1, -1, 1, -1, 1) is an energy minimum. If the network is given the input state (1, -1, -1, -1, 1), it will converge to (1, -1, 1, -1, 1). A network is properly trained when the energy of states which the network should remember are local minima. 

A learning rule for a Hopfield Network should have both of the following two properties:

- *Local*: A learning rule is *local* if each weight is updated using information available to neurons on either side of the connection that is associated with that particular weight.
- *Incremental*: New patterns can be learned without using information from the old patterns that have been also used for training. That is, when a new pattern is used for training, the new values for the weights only depend on the old values and on the new pattern. A learning system that is not incremental is trained only once with a huge batch of training data.

The following rules are local and incremental.

### Hebbian learning rule for Hopfield networks

[Hebbian Theory](https://en.wikipedia.org/wiki/Hebbian_theory) (Donald Hebb, 1949), was proposed to explain "associative learning" where simultaneous activation of neuron cells leads to pronounced increases in synaptic strength between those cells. It is often summarized as "Neurons that fire together, wire together. Neurons that fire out of sync, fail to link".

The Hebbian rule for learning Hopfield Networks is implemented in the following manner when learning $n$ binary patterns:

$w_{ij}=\frac {1}{n} \sum _{\mu =1}^{n} \epsilon _{i}^{\mu } \epsilon _{j}^{\mu }$

where $\epsilon _{i}^{\mu }$ represents bit ```i``` from pattern $\mu$.

If the bits corresponding to neurons ```i``` and ```j``` are equal in pattern $\mu$, then the product $\epsilon_{i}^{\mu } \epsilon_{j}^{\mu }$ will be positive. This would, in turn, have a positive effect on the weight $w_{ij}$ and the values of ```i``` and ```j``` will tend to become equal. The opposite happens if the bits corresponding to neurons ```i``` and ```j``` are different.

### The Storkey learning rule

Storkey (1997) discovered a learning rule that showed a greater capacity to train a Hopfield than a corresponding network trained using the Hebbian rule. The weight matrix of an [attractor neural network](https://en.wikipedia.org/wiki/Attractor_network) is said to follow the Storkey learning rule if it obeys:

$w_{ij}^{\nu } = w_{ij}^{\nu -1} + \frac{1}{n}\epsilon_{i}^{\nu} \epsilon_{j}^{\nu} - \frac{1}{n}\epsilon_{i}^{\nu} h_{ji}^{\nu} - \frac{1}{n} \epsilon_{j}^{\nu }h_{ij}^{\nu}$

where $h_{ij}^{\nu} = \sum _{k=1~:~i\neq k\neq j}^{n}w_{ik}^{\nu -1}\epsilon _{k}^{\nu }$ is a form of *local field* at neuron ```i```.

This learning rule is local since the synapses take into account only neurons at their sides. The rule makes use of more information from the patterns and weights than the generalized Hebbian rule, due to the effect of the local field.

That I can tell, there are no current implementations of Hopfield networks for modern networks.  However, their probabilistic counterpart the Boltzmann machine does present some opportunities.

### References

*Hopfield, John J. “Neural networks and physical systems with emergent collective computational abilities.” Proceedings of the national academy of sciences 79.8 (1982): 2554-2558.* [https://bi.snu.ac.kr/Courses/g-ai09-2/hopfield82.pdf](https://bi.snu.ac.kr/Courses/g-ai09-2/hopfield82.pdf).

*Wikipedia. Hopfield Network. Accessed June 2018. [https://en.wikipedia.org/wiki/Hopfield_network](https://en.wikipedia.org/wiki/Hopfield_network).*

*Donald Olding Hebb. The organization of behavior. A neuropsychological theory. John Wiley and Sons, 1949. [http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf).*

*Amos Storkey. Increasing the capacity of a Hopfield network without sacrificing functionality. ICANN'97. 1997. [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.103&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.103&rep=rep1&type=pdf).*

# Boltzmann Machine (BM)

<img src="https://cdn-images-1.medium.com/max/750/1*DXlFPJ1CqE-YuvqNJ0cnpw.png" width="400px">

Boltzmann machines are very similar to HNs where some cells are marked as input and remain hidden. Input cells become output as soon as each hidden cell update their state (during training, BMs / HNs update cells one by one, and not in parallel).

This is the first network topology that was succesfully tained using [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) approach.

Multiple stacked Boltzmann Machines can for a so-called [Deep belief network](https://en.wikipedia.org/wiki/Deep_belief_network) (see below), that is used for feature detection and extraction.

# Restricted Boltzmann Machine (RBM)

<img src="https://cdn-images-1.medium.com/max/750/1*nWB337Vo0kukMkWISNwYcA.png" width="400px">

RBMs resemble, in the structure, BMs but, due to being restricted, allow to be trained using backpropagation just as FFs (with the only difference that before backpropagation pass data is passed back to input layer once). 

### References

*Hengyuan Hu, Lisheng Gao, Quanbin Ma. Deep Restricted Boltzmann Networks. November 2016. Accessed June 2018. [https://arxiv.org/pdf/1611.07917.pdf](https://arxiv.org/pdf/1611.07917.pdf).*



# Perceptrons

<img src="https://cdn-images-1.medium.com/max/750/1*JLlHNhYjyY9h9y1D1sy8Zw.png" width="200px">

*Perceptrons* are the oldest (1957) and simplest model of a Neuron. A single perceptron is capable of the simplest of all classification problems - dividing the population with a straight line and in themselves are very interesting for more general problems.

## Neuron - The Basic Neural Network Cell

<img src="http://www.asimovinstitute.org/wp-content/uploads/2016/12/ffcell.png">

The basic neural network cell is connected to other neurons via weights, that is, it can be connected to all the neurons in the previous layer. Each connection has its own weight, which is often just a random number at first. A weight can be negative, positive, very small, very big or zero. The value of each of the cells it’s connected to is multiplied by its respective connection weight. The resulting values are all added together. On top of this, a bias is also added. A bias can prevent a cell from getting stuck on outputting zero and it can speed up some operations, reducing the amount of neurons required to solve a problem. The bias is also a number, sometimes constant (often -1 or 1) and sometimes variable. This total sum is then passed through an activation function, the resulting value of which then becomes the value of the cell. 

#### References

*Rosenblatt, Frank. “The perceptron: a probabilistic model for information storage and organization in the brain.” Psychological review 65.6 (1958): 386.* 

*Perceptron. Wikipedia. Accessed June 2018, [https://en.wikipedia.org/wiki/Perceptron](https://en.wikipedia.org/wiki/Perceptron).*

*Geoffrey Hinton. Neural Networks for Machine Learning. Coursera. [https://www.coursera.org/learn/neural-networks](https://www.coursera.org/learn/neural-networks).*

*Andrew Ng. Neural Networks and Deep Learning. Coursera. [https://www.coursera.org/learn/neural-networks-deep-learning](https://www.coursera.org/learn/neural-networks-deep-learning).*

# Feed Forward Networks

<img src="https://cdn-images-1.medium.com/max/750/1*W9awL0IGeXBcrPAxQPtbew.png" width="200px">

*Feed forward neural networks* were first developed in 1958. For much of its early history, feed forward networks were only feed forward. Minsky and Papert's book [Perceptrons](https://archive.org/details/Perceptrons) (1969) stated that very simple feed forward networks with only an input and output layer could perform only the most simple tasks which caused most researcher's to give up on with one notable exception: Geoffrey Hinton.  In 1986, [Rumelhart, Hinton, and Williams](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) showed that back propagation can be used to train neural networks. In some sense, it was Hinton, along with two of his students Alex Krizhevsky and Ilya Sutskever (both leaders in the field) who created the modern machine learning renaissance with their 2012 paper [ImageNet Classification with Deep Convolutional Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) which implemented back propagated networks which beat the best image recognition systems of the time.

Generally, these simple networks follow the following rules:

1. All nodes are in layers fully connected with basic neural cells.  Fully connected layers are often referred to as dense layers.
2. Activation flows from input layer to output, without back loops also known as cycles.  In other words, feed forward networks work on an acyclic graph.
3. There is one layer between input and output known as the hidden layer.

The simplest somewhat practical network has two input cells and one output cell, which can be used to model logic gates.

After 2012, these networks are typically trained using [backpropagation](https://en.wikipedia.org/wiki/Backpropagation). Backpropagation is used in supervised learning.  The error being back-propagated is often some variation of the difference between the input and the output (like MSE or just the linear difference). If a network has enough hidden neurons, it can theoretically always model the relationship between the input and output (Universal Approximation Theorem by Csáji).

> **Supervised and Unsupervised Learning**.  In *supervised learning*, networks are given labeled datasets: paired information consisting of a set of numbers (“what goes in”) and what the number represents which is the label (“what we want to have coming out”). In *unsupervised learning*, the network is given only the input set of numbers and let the network determines relationships or patterns in the data. 

Practically, their use is a lot more limited but they are popularly combined with other networks to form new networks. 

A nice, but rather formal walkthrough of Feed Forward networks is provided by [John McGonagle](https://brilliant.org/wiki/feedforward-neural-networks/).

## Keras Example

This [notebook](https://nbviewer.jupyter.org/github/jeffreyrnorton/MachineLearningNotes/blob/master/Feed_Forward_Network.ipynb) gives an example of a feed forward network and explores some different configurations and considerations.  These topics will be discussed further.

### References

*Marvin L. Minsky, Seymour A. Papert. Perceptrons. 1969. [https://archive.org/details/Perceptrons](https://archive.org/details/Perceptrons).*

*David E. Rumelhart, Geoffrey E. Hinton & Ronald J. Williams. Learning representations by back-propagating errors. Nature. October 1986. [https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf).*

*Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. ImageNet Classification with Deep Convolutional Networks. 2012. [https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).*

*G. Cybenko. Approximation of Superpositions of a Sigmoidal Function. Mathematics of Control, Signals, and Systems. 1989. http://www.dartmouth.edu/~gvc/Cybenko_MCSS.pdf.*

*Balázs Csanád Csáji. Approximation with Artificial Neural Networks; Faculty of Sciences; Eötvös Loránd University, Hungary. 2001. [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.2647&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.2647&rep=rep1&type=pdf).*

*Michael Nielsen. A visual proof that neural nets can compute any function. Chapter 4. Neural Networks and Deep Learning. Accessed June 2018. [http://neuralnetworksanddeeplearning.com/chap4.html](http://neuralnetworksanddeeplearning.com/chap4.html).*

# Radial Basis Function Network (RBFN)

<img src="https://cdn-images-1.medium.com/max/750/1*ywIq8moPGBrQDf_dq71dhQ.png" width="275px">

**RBF neural networks** are FF (feed forward) neural networks that use [radial basis function](https://en.wikipedia.org/wiki/Radial_basis_function) as activation function instead of [logistic function](https://en.wikipedia.org/wiki/Logistic_function).

The Logistic function maps some arbitrary value to a 0…1 range, answering a “yes or no” question. It is good for classification and decision making systems, but does not work well for continuous values.

On the other hand, radial basis functions answer the question “how far are we from the target”? This is perfect for function approximation, and machine control (as a replacement of [PID](https://en.wikipedia.org/wiki/PID_controller) controllers, for example).

To be short, these are just Feed Forward networks with different activation function and appliance.

In Keras, radial basis functions are not provided. However, Keras provides a way to define customized activation functions ([Stackoverflow](https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras)).  Consider writing an activation function for the Gaussian: $$\phi \left(r\right)=e^{-\left(\varepsilon r\right)^{2}}$$.

```python
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def gaussian(r, epsilon=0.01):
    return math.exp(-(epsilon*r)**2)

get_custom_objects().update({'custom_activation': Activation(gaussian)})

model.add(Activation(gaussian))
```

However, a more comprehensive approach is that taken by [Petra Vidnerova](https://github.com/PetraVidnerova/rbf_keras) who writes her own layer in Keras.  I consider writing your own layers a critical skill and have noticed that Keras is very extensible in this fashion.  Here is an [a notebook demonstrating an RBFN](https://nbviewer.jupyter.org/github/jeffreyrnorton/MachineLearningNotes/blob/master/RadialBasisFunctionNetwork.ipynb) based on Petra's work.

Note: [McCormicks's Paper](http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/) looks good on paper, but I had trouble getting it to work.  I think the problem is that he is still thinking of RBF's as a way to solve classification problems when really, they are met to be a way to fit functions.

### References

*Broomhead, David S., and David Lowe. Radial basis functions, multi-variable functional interpolation and adaptive networks. No. RSRE-MEMO-4148. ROYAL SIGNALS AND RADAR ESTABLISHMENT MALVERN (UNITED KINGDOM), 1988. [http://www.dtic.mil/cgi-bin/GetTRDoc?AD=ADA196234](http://www.dtic.mil/cgi-bin/GetTRDoc?AD=ADA196234).*

*Vidnerova, Petra. rbf_keras. Github. Accessed June 2018. [https://github.com/PetraVidnerova/rbf_keras](https://github.com/PetraVidnerova/rbf_keras).*



# DNNs, Deep Neural Nets, or Deep Feed Forward Networks

<img src="https://cdn-images-1.medium.com/max/750/1*Vyz8w8ZGiZCl17birZqIyw.png" width="275px">

**DFF neural networks** go by several names: Deep Neural Networks (DNNs), Artificial Neural Networks (ANNs), opened the pandora box of [deep learning](https://en.wikipedia.org/wiki/Deep_learning) in early 90s. These are Feed Forward Neural Networks, but with more than one hidden layer. Why are they so different?

When training a traditional FF, we pass a small amount of error to previous layer. However, stacking more layers led to exponential growth of training times in order to drive down the error accumulation, making DFFs quite impractical. In the early 2000s, several approaches were developed that allowed to train DFFs effectively; now they form a core of modern Machine Learning systems, covering the same purposes as FFs, but with much better results.

## Topology of Deep Neural Networks (DNN)

Typically, ANNs and DNNs are composed of dense layers.  ANNs or Feed Forward networks are typically defined by having one hidden layer and DNNs have two or more.

The topology or architecture of a neural network is defined as the type and number of layers and the type and number of neurons in each layer.  By following a small set of clear rules, one can programmatically configure a competent network architecture. Following this schema will generate a competent architecture but probably not an optimal one. Once this network is initialized, the configuration can be iteratively tuned during training using a number of ancillary algorithms including pruning, genetic optimization, etc.

Every neural network has three types of layers: *input*, *hidden*, and *output*..

### The Input Layer

Every neural network should have exactly one.  With respect to the number of neurons comprising this layer, this parameter is completely and uniquely determined once the shape of the training data is known. Specifically, the number of neurons comprising the hidden layer is equal to the number of features (columns) in the data. Some neural network configurations add one additional node for a bias term.

### The Output Layer

Every neural network has exactly one output layer. Determining its size (number of neurons) is simple; it is completely determined by the chosen model configuration.

Is your neural network going to run a classification problem (returns a discrete value) or a regression problem (returns a continuous value)?

* If the neural network is a regressor, then the output layer has a single node.

* If the neural network is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.

### The Hidden Layers

How many hidden layers should be in the neural network?

If your data is linearly separable, then you don't need any hidden layers at all. Of course, you don't need an neural network to resolve your data either, but it will still do the job.

There is a mountain of comentary on the question of hidden layer configuration in neural networks (see Sarle's FAQ for example). One issue is the performance difference from adding additional hidden layers. One hidden layer is sufficient for a large majority of classification and regression problems as it can approximate any (reasonable) function given enough training data.

However, there are a few difficulties with using an extremely wide, shallow network. Very wide, shallow networks are very good at memorization, but not so good at generalization. If you train the network with *every* possible input value, a super wide network could eventually memorize the corresponding output value that you want. But that is not useful because for any practical application you will not have every possible input value to train with.

Jeff Heaton (Heaton Research) provides these outlines for choosing the number of hidden layers:

**Table: Determining the Number of Hidden Layers**

| **Num   Hidden Layers** | **Result**                                                   |
| ----------------------- | ------------------------------------------------------------ |
| none                    | Only capable of representing linear separable functions or decisions. |
| 1                       | Can  approximate any function that contains a continuous mapping from one finite space to another. |
| 2                       | Can  represent an arbitrary decision boundary to arbitrary accuracy with rational activation functions and can approximate any smooth mapping to any accuracy. |
| >2                      | Additional layers can learn complex representations (sort of automatic feature engineering) for layer layers. |

Another way of looking at the power of the additional layers, is that they can learn features at various levels of abstraction. For example, with a deep convolutional neural network used to classify images, the first layer will train itself to recognize very basic things like edges, the next layer will train itself to recognize collections of edges such as shapes, the next layer will train itself to recognize collections of shapes like eyes or noses, and the next layer will learn even higher-order features like faces. Multiple layers are much better at generalizing because they learn all the intermediate features between the raw data and the high-level classification. That explains why you might use a deep network rather than a very wide but shallow network.

Why not use a very deep, very wide network? One possible answer is that the network should be as small as possible and still produce good results. As you increase the size of the network, you introducing more parameters that your network needs to learn and run the risk of increasing the chances of overfitting. If you build a very wide, very deep network, you run the chance of each layer just memorizing what you want the output to be, and you end up with a neural network that fails to generalize to new data.

In either case (wide and shallow or deep and narrow), you have many more parameters to learn and it will take longer to train. 

Deep networks already can be very computationally expensive to train, so there's a strong incentive to make them wide enough that they work well, but no wider; that is, how few neurons are sufficient in the hidden layers? Jeff Heaton gives some empirically-derived rules-of-thumb.

* The number of hidden neurons should be between the size of the input layer and the size of the output layer.
* The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
* The number of hidden neurons should be less than twice the size of the input layer.

### Optimization of the Network Configuration

Pruning describes a set of techniques to trim network size (by nodes not layers) to improve computational performance and sometimes also resolution performance. The gist of these techniques is removing nodes from the network during training by identifying those nodes which, if removed from the network, would not noticeably affect network performance (i.e., resolution of the data). 

Even without using a formal pruning technique, you can get a rough idea of which nodes are not important by looking at your weight matrix after training; the nodes with weights very close to zero--it's the nodes on either end of those weights that are often removed during pruning.) Obviously, if you use a pruning algorithm during training then begin with a network configuration that is more likely to have excess (i.e., *prunable*) nodes--in other words, when deciding on a network architecture, err on the side of more neurons, if you add a pruning step. Put another way, by applying a pruning algorithm to your network during training, you can approach optimal network configuration.

Another approach that has been studied since the 1990s is to apply a genetic algorithm to optimize the network topology.  It is still an active area of study - see the paper by Idrissi et al.

### Example

The TensorFlow playground provides the difficult spiral data model which we can attempt to solve.  First, we show underfitting. We can see that while the convergence is smooth, it is slow and the limited number of neurons in the hidden layer cannot capture all the features.

![](https://raw.githubusercontent.com/jeffreyrnorton/GoogleCloudPlatformNotes/master/Images//Spiral_Underfit.jpg)

Erring to the other side, we see that while this configuration with the same number of neurons as the input layer appears to converge, we also note that the convergence is slow.  We see that the convergence is a "bumpy trip" and finally, if we look at the fifth node down, we can see that it appears to be capturing very little of the feature.

![](https://raw.githubusercontent.com/jeffreyrnorton/GoogleCloudPlatformNotes/master/Images//Spiral_Overfit.jpg)

If we look at the nodes in the network below, it appears that the top node has very little influence in the model.  Yet, if we run four nodes, we get terrible convergence - that is, a bumpy ride to minimization.  However, with five nodes (even with the top one appearing to provide much in the way of feature recognition) actually captures the features in less than have of the epochs.

![](https://raw.githubusercontent.com/jeffreyrnorton/GoogleCloudPlatformNotes/master/Images//Spiral_JustRight.jpg)

### Expanding Feed Forward networks to Deep Nets

In this [notebook](https://nbviewer.jupyter.org/github/jeffreyrnorton/MachineLearningNotes/blob/master/FeedForward2DeepNetwork.ipynb) we explore how adding extra hidden layers and dropout which we discuss below affect our solution with the diabetes study.  A naïve approach assumes that just adding new layers makes things better, but that is not always the case.

### References

Cross Validated, "How to choose the number of hidden layers and nodes in a feedforward neural network", March 15, 2017, https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

Cross Validated, "Why are neural networks becoming deeper, but not wider?", July 13, 2016, https://stats.stackexchange.com/questions/222883/why-are-neural-networks-becoming-deeper-but-not-wider

Heaton Research, "The Number of Hidden Layers", June 1, 2017, http://www.heatonresearch.com/2017/06/01/hidden-layers.html

Idrissi, J, et al, Genetic Algorithm for Neural Network Architecture Optimization, 2016, https://www.researchgate.net/profile/Mohammed_Amine_Janati_Idrissi/publication/309694276_Genetic_algorithm_for_neural_network_architecture_optimization/links/59f9f7dbaca272026f6ecab8/Genetic-algorithm-for-neural-network-architecture-optimization.pdf

Sarle, W.S., comp.ai.neural-nets FAQ, 2002, http://www.faqs.org/faqs/ai-faq/neural-nets/part1/preamble.html

Stathakis, D., "How many hidden layers and nodes", International Journal of Remote Sensing
Vol. 30, No. 8, 20 April 2009, 2133–2147, http://dstath.users.uth.gr/papers/IJRS2009_Stathakis.pdf

TensorFlow Playground, Accessed May 10, 2018, http://playground.tensorflow.org

# Regularization and Dropout
## Underfitting and Overfitting
Before we can talk about regularization and dropout, it is helpful to do a brief review of underfitting and overfitting.

# RNNs
<img src="https://cdn-images-1.medium.com/max/750/1*mC-OU9esw8BgE7H23W56-w.png" width="400px">

**Recurrent Neural Networks** introduce different type of cells — Recurrent cells. The first network of this type was so called [Jordan network](https://en.wikipedia.org/wiki/Recurrent_neural_network#Jordan_network), when each of hidden cell received it’s own output with fixed delay — one or more iterations. Apart from that, it was like common FNN.

Of course, there are many variations — like passing the state to input nodes, variable delays, etc, but the main idea remains the same. This type of NNs is mainly used then *context* is important — when decisions from past iterations or samples can influence current ones. The most common examples of such contexts are texts — a word can be analyzed only in context of previous words or sentences.

### Resources

Fei-Fei Li, Justin Johnson & Serea Young. Lecture 9: Recurrent Neural Networks. CS 231n, Stanford University, 2017. [http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf).

Abhishek Narwekar, Anusri Pampari. Recurrent Neural Network Architectures, CS 598: Deep Learning and Recognition. University of Illinois. Fall 2016. [http://slazebni.cs.illinois.edu/spring17/lec20_rnn.pdf](http://slazebni.cs.illinois.edu/spring17/lec20_rnn.pdf).

Ian Goodfellow, Yoshua Bengio and Aaron Courville. "Chapter 10: Sequence Modeling: Recurrent and Recursive Nets". Deep Learning. MIT Press. 2016. [http://www.deeplearningbook.org/contents/rnn.html](http://www.deeplearningbook.org/contents/rnn.html).

Parag Mital. Session 4: Visualizing and Hallucinating Representations. Creative Applications of Deep Learning, Kadenze. Last accessed 2018. [https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/sessions/visualizing-and-hallucinating-representations](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/sessions/visualizing-and-hallucinating-representations).

DL4J. A Beginner's Guide to Recurrent Networks and LSTMs. [https://deeplearning4j.org/lstm](https://deeplearning4j.org/lstm).

DL4J. Tutorial: Recurrent Networks and LSTMs. [https://deeplearning4j.org/recurrentnetwork](https://deeplearning4j.org/recurrentnetwork).

Andrej Karpathy. The Unreasonable Effectiveness of Recurrent Neural Networks. May 21, 2015. [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

## LSTMs

<img src="https://cdn-images-1.medium.com/max/750/1*z380CENiF6G5NcQuXrw-hw.png" width="375px">

This type introduces a *memory cell,* a special cell that can process data when data have time gaps (or lags). RNNs can process texts by “keeping in mind” ten previous words, and LSTM networks can process video frame “keeping in mind” something that happened many frames ago. LSTM networks are also widely used for writing and speech recognition.

*Memory cells* are actually composed of a couple of elements — called gates, that are recurrent and control how information is being remembered and forgotten. The structure is well seen in the wikipedia illustration (note that there are no activation functions between blocks):

[**Long short-term memory - Wikipedia**
*Long short-term memory ( LSTM) is an artificial neural network architecture that supports machine learning. It is…*en.wikipedia.org](https://en.wikipedia.org/wiki/Long_short-term_memory#/media/File:Peephole_Long_Short-Term_Memory.svg)

The (x) thingies on the graph are *gates*, and they have they own weights and sometimes activation functions. On each sample they decide whether to pass the data forward, erase memory and so on — you can read a quite more detailed explanation [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). Input gate decides how many information from last sample will be kept in memory; output gate regulate the amount of data passed to next layer, and forget gates control the tearing rate of memory stored.

This is, however, a very simple implementation of LSTM cells, many others architectures exist.

# Gated Recurrent Network (GRU)

<img src="https://cdn-images-1.medium.com/max/750/1*UmLdrVH-_EMnL-pUjcYKhA.png" width="400px">



GRUs are LSTMs with different gating. Period.

Sounds simple, but lack of output gate makes it easier to repeat the same output for a concrete input multiple times, and are currently used the most in sound (music) and speech synthesis.

The actual composition, though, is a bit different: all LSTM gates are combined into so-called *update gate*, and *reset gate* is closely tied to input.

They are less resource consuming than LSTMs and almost the same effectiveness.

# Auto Encoder (AE)

<img src="https://cdn-images-1.medium.com/max/750/1*OqWtq4tg564vLxNbzSIsfA.png" width="400px">

Autoencoders are used for classification, clustering and feature compression.

When you train FF neural networks for classification you mostly must feed then X examples in Y categories, and expect one of Y output cells to be activated. This is called “supervised learning”.

AEs, on the other hand, can be trained without supervision. Their structure — when number of hidden cells is smaller than number of input cells (and number of output cells equals number of input cells), and when the AE is trained the way the output is as close to input as possible, forces AEs to generalize data and search for common patterns.

# Variatonal Auto Encoder (VAE)

<img src="https://cdn-images-1.medium.com/max/750/1*azeRzbuEXrHAa6VgY8Oa5Q.png" width="400px">

VAEs, comparing to AE, compress probabilities instead of features.

Despite that simple change, when AEs answer the question “how can we generalise data?”, VAEs answer the question “how strong is a connection between two events? should we distribute error between the two events or they are completely independent?”.

A little bit more in-depth explanation (with some code) is accessible [here](https://github.com/kvfrans/variational-autoencoder).

# Denoising Auto Encoder (DAE)

<img src="https://cdn-images-1.medium.com/max/750/1*iXr_UJKiFxf0u7gE305u8Q.png" width="400px">

While AEs are cool, they sometimes, instead of finding the most robust features, just adapt to input data (it is actually an example of overfitting).

DAEs add a bit of noise on the input cells — vary the data by random bit, randomly switch bits in input, etc. By doing that, one forces DAE to reconstruct output from a bit noisy input, making it more general and forcing to pick more common features.

# Sparse Auto Encoder (SAE)

<img src="https://cdn-images-1.medium.com/max/750/1*6JX4x6oJp5fUseiARDwXZg.png" width="400px">

SAE is yet another autoencoder type that in some cases can reveal some hidden grouping patters in data. Structure is the same as in AE but hidden cell count is bigger than input / output layer cell count. 

------

## Natural Language Processing (NLP)

DL4J. Word2Vec, Doc2vec & GloVe: Neural Word Embeddings for Natural Language Processing. [https://deeplearning4j.org/word2vec](https://deeplearning4j.org/word2vec).

Jeffrey Pennington, Richard Socher, Christopher D. Manning. GloVe: Global Vectors for Word Representation . https://nlp.stanford.edu/projects/glove/.





# Deep Belief Networks

<img src="https://cdn-images-1.medium.com/max/750/1*6S8uHAzhWi0wQQ_jUBq0zA.png" width="400px">

DBNs, mentioned above, are actually a stack of Boltzmann Machines (surrounded by VAEs). They can be chained together (when one NN trains another) and can be used to generate data by already learned pattern. 

# CNNs, ConvNets, or Convolutional Nets

<img src="https://cdn-images-1.medium.com/max/750/1*L1NzxHnhto2hn1TaDRoudQ.png" width="500px">

Given the number of layers, we can also refer to these as deep convolutional nets (DCNs).  DCNs nowadays are stars of artificial neural networks. They feature convolution cells (or pooling layers) and kernels, each serving a different purpose.

Convolution kernels actually process input data, and pooling layers simplify it (mostly using non-linear functions, like max), reducing unnecessary features.

Typically used for image recognition, they operate on small subset of image (something about 20x20 pixels). The input window is sliding along the image, pixel by pixel. The data is passed to convolution layers, that form a funnel (compressing detected features). From the terms of image recognition, first layer detects gradients, second lines, third shapes, and so on to the scale of particular objects. DFFs are commonly attached to the final convolutional layer for further data processing.

## AlexNet

**Full (simplified) AlexNet architecture:**

[227x227x3] INPUT
[55x55x96] CONV1: 96 11x11 filters at stride 4, pad 0
[27x27x96] MAX POOL1: 3x3 filters at stride 2
[27x27x96] NORM1: Normalization layer
[27x27x256] CONV2: 256 5x5 filters at stride 1, pad 2
[13x13x256] MAX POOL2: 3x3 filters at stride 2
[13x13x256] NORM2: Normalization layer
[13x13x384] CONV3: 384 3x3 filters at stride 1, pad 1
[13x13x384] CONV4: 384 3x3 filters at stride 1, pad 1
[13x13x256] CONV5: 256 3x3 filters at stride 1, pad 1
[6x6x256] MAX POOL3: 3x3 filters at stride 2
[4096] FC6: 4096 neurons
[4096] FC7: 4096 neurons
[1000] FC8: 1000 neurons (class scores)

**Details/Retrospectives:**

- first use of ReLU
- used Norm layers (not common anymore)
- heavy data augmentation
- dropout 0.5
- batch size 128
- SGD Momentum 0.9
- Learning rate 1e-2, reduced by 10
  manually when val accuracy plateaus
- L2 weight decay 5e-4
- 7 CNN ensemble: 18.2% -> 15.4%

![](https://cdn-images-1.medium.com/max/1000/0*xPOQ3btZ9rQO23LK.png)




## VGG

Keras VGG 16 pre-loaded model: [https://keras.io/applications/#vgg16](https://keras.io/applications/#vgg16).

Keras VGG 19 pre-loaded model: [https://keras.io/applications/#vgg19](https://keras.io/applications/#vgg19).

## GoogLeNet

## ResNet

## Xception

Xception pre-loaded model: [https://keras.io/applications/#xception](https://keras.io/applications/#xception).

### Resources

Siddarth Das. CNNs Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet and more …. Nov 16, 2017. Medum.com. [https://medium.com/@siddharthdas_32104/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5](https://medium.com/@siddharthdas_32104/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5).

Fei-Fei Li, Justin Johnson & Serea Young. Lecture 9: CNN Architectures. CS 231n, Stanford University, 2017. [http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf).

Fei-Fei Li, Justin Johnson & Serea Young. Lecture 11: Detection and Segmentation. CS 231n, Stanford University, 2017. [http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf).

Fei-Fei Li, Justin Johnson & Serea Young. Lecture 12: Visualization and Understanding. CS 231n, Stanford University, 2017. [http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf).

Jonathan Long. Understanding and Designing Convolutional Neural Networks for Local Recognition. Technical Report No. UCB/EECS-2016-97. May 13, 2016. [https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-97.pdf](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-97.pdf).

Ian Goodfellow, Yoshua Bengio and Aaron Courville. "Chapter 9: Convolutional Networks". Deep Learning. MIT Press. 2016. [http://www.deeplearningbook.org/contents/convnets.html](http://www.deeplearningbook.org/contents/convnets.html).

Christopher Olah. Conv Nets: A Modular Perspective. July 8, 2014. http://colah.github.io/posts/2014-07-Conv-Nets-Modular/.

Christopher Olah. Understanding Convolutions. July 13, 2014. [http://colah.github.io/posts/2014-07-Understanding-Convolutions/](http://colah.github.io/posts/2014-07-Understanding-Convolutions/).

Christopher Olah. Groups & Group Convolutions. December 8, 2014. [http://colah.github.io/posts/2014-12-Groups-Convolution/](http://colah.github.io/posts/2014-12-Groups-Convolution/).

DL4J. A Beginner's Guide to Deep Convolutional Neural Networks. [https://deeplearning4j.org/convolutionalnetwork](https://deeplearning4j.org/convolutionalnetwork).

# Deconvolutional Network (DNs)

<img src="https://cdn-images-1.medium.com/max/750/1*P4k0hiiaUbsI6tss_QjkrQ.png" width="500px">

DNs are DCNs reversed. DN takes cat image, and produces vector like { dog: 0, lizard: 0, horse: 0, cat: 1 }. DCN can take this vector and draw a cat image from that. I tried to find a solid demo, but the best demo is on [youtube](https://www.youtube.com/watch?v=rAbhypxs1qQ). 

// Look at the Kadenze stuff //

# Deep Convolutional Inverse Graphics Models (DCIGN)

<img src="https://cdn-images-1.medium.com/max/750/1*l5opPhyeIZdMJPzaXfb5rg.png" width="600px">

DCIGN looks like DCN and DN glued together, but it is not quire correct.

Actually, it is an autoencoder. DCN and DN do not act as separate networks, instead, they are spacers for input and output of the network. Mostly used for image processing, these networks can process images that they have not been trained with previously. These nets, due to their abstraction levels, can remove certain objects from image, re-paint it, or replace horses with zebras like the famous [CycleGAN](https://github.com/junyanz/CycleGAN)did.

# Generative Adversarial Networks (GAN)

<img src="https://cdn-images-1.medium.com/max/750/1*6I157u13ns-2DIfl7mpXDQ.png" width="500px">

### Resources

Fei-Fei Li, Justin Johnson & Serea Young. Lecture 13: Generative Models. CS 231n, Stanford University, 2017. [http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf).

Ian Goodfellow and Yoshua Bengio and Aaron Courville. "Chapter 20: Generative Models". Deep Learning. MIT Press. 2016. [http://www.deeplearningbook.org/contents/generative_models.html](http://www.deeplearningbook.org/contents/generative_models.html).

Yoshua Bengio. "Generative Deep Models". IFT 6266, University of Montreal. 2016. [http://www.iro.umontreal.ca/~bengioy/ift6266/H16/generative.pdf](http://www.iro.umontreal.ca/~bengioy/ift6266/H16/generative.pdf).

Parag Mital. Session 5: Generative Models. Creative Applications of Deep Learning, Kadenze. Last accessed 2018. [https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/sessions/generative-models](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/sessions/generative-models).

DL4J. GAN: A Beginner's Guide to Generative Adversarial. [https://deeplearning4j.org/generative-adversarial-network](https://deeplearning4j.org/generative-adversarial-network).

Rowel Atienza, GAN by Example using Keras on Tensorflow Backend, Towards Data Science. March 29, 2017. [https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0).

# Kohonen Networks (KN)

<img src="http://www.asimovinstitute.org/wp-content/uploads/2016/09/kn.png" width="400px">

Kohonen networks are also known as self organizing (feature) maps, SOM, SOFM. KNs utilise competitive learning to classify data without supervision. Input is presented to the network, after which the network assesses which of its neurons most closely match that input. These neurons are then adjusted to match the input even better, dragging along their neighbours in the process. How much the neighbours are moved depends on the distance of the neighbours to the best matching units. KNs are sometimes not considered neural networks either.

*Kohonen, Teuvo. “Self-organized formation of topologically correct feature maps.” Biological cybernetics 43.1 (1982): 59-69. [http://cioslab.vcu.edu/alg/Visualize/kohonen-82.pdf](http://cioslab.vcu.edu/alg/Visualize/kohonen-82.pdf).

# Reinforcement Learning

### Resources

Fei-Fei Li, Justin Johnson & Serea Young. Lecture 14: Reinforcement Learning. CS 231n, Stanford University, 2017. [http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf).




