# Probabilistic Graphical Models



## Markov Chains

![](http://www.asimovinstitute.org/wp-content/uploads/2016/09/mc.png)

### Background

**Markov chains (MC or discrete time Markov Chain, DTMC)** are kind of the predecessors to BMs and HNs. They can be understood as follows: from this node where I am now, what are the odds of me going to any of my neighboring nodes? They are memoryless (i.e. Markov Property) which means that every state you end up in depends completely on the previous state. While not really a neural network, they do resemble neural networks and form the theoretical basis for BMs and HNs. MC aren’t always considered neural networks, as goes for BMs, RBMs and HNs. Markov chains aren’t always fully connected either.

*Hayes, Brian. “First links in the Markov chain.” American Scientist 101.2 (2013): 252.* [https://www.americanscientist.org/article/first-links-in-the-markov-chain](https://www.americanscientist.org/article/first-links-in-the-markov-chain).

### Use Cases

#### PageRank

#### MCMC - Markov Chain Monte Carlo



### Example



## Boltzmann Machines

![](http://www.asimovinstitute.org/wp-content/uploads/2016/09/bm.png)

**Boltzmann machines (BM)** are a lot like HNs, but: some neurons are marked as input neurons and others remain “hidden”. The input neurons become output neurons at the end of a full network update. It starts with random weights and learns through back-propagation, or more recently through contrastive divergence (a Markov chain is used to determine the gradients between two informational gains). Compared to a HN, the neurons mostly have binary activation patterns. As hinted by being trained by MCs, BMs are stochastic networks. The training and running process of a BM is fairly similar to a HN: one sets the input neurons to certain clamped values after which the network is set free (it doesn’t get a sock). While free the cells can get any value and we repetitively go back and forth between the input and hidden neurons. The activation is controlled by a global temperature value, which if lowered lowers the energy of the cells. This lower energy causes their activation patterns to stabilise. The network reaches an equilibrium given the right temperature.

*Hinton, Geoffrey E., and Terrence J. Sejnowski. “Learning and releaming in Boltzmann machines.” Parallel distributed processing: Explorations in the microstructure of cognition 1 (1986): 282-317.*
[https://www.researchgate.net/profile/Terrence_Sejnowski/publication/242509302_Learning_and_relearning_in_Boltzmann_machines/links/54a4b00f0cf256bf8bb327cc.pdf](https://www.researchgate.net/profile/Terrence_Sejnowski/publication/242509302_Learning_and_relearning_in_Boltzmann_machines/links/54a4b00f0cf256bf8bb327cc.pdf)

## Restricted Boltzmann Machines

![](http://www.asimovinstitute.org/wp-content/uploads/2016/09/rbm.png)

