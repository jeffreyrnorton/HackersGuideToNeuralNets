---
title: "Neurons"
author: "Jeffrey Norton"
date: "January 2017"
output:
  html_document:
    keep_md: yes
---



# McCullogh-Pitts Neurons

There are several references which are helpful in understanding McCullogh-Pitts neurons,
but [this one](https://www.lri.fr/~marc/EEAAX/Neurones/tutorial/mcpits/html/index.html)
was helpful.  The key formula is given by Hinton in his PDF notes.

Consider a set of inputs sent to a McCullogh-Pitt neuron as shown in the following figure:

![](https://hako.github.io/dissertation/figures/figure_1.svg)

The activation function

![](https://github.com/jeffreyrnorton/NeuralNetworking/blob/master/1_Neurons/Theshold.PNG?raw=true)

is given as follows:

1. Compute a weighted sum plus a bias: $z = b + \sum_i x_i w_i$.  
2. Output 1 if $z > 0$, specifically $y = 1$ if $z \gt 0$, $0$ otherwise

## Example
The inputs $x$ consist of excitatory inputs with values ${0, 1}$,

```r
n=10
(x_excite=sample(c(0,1), n, replace=TRUE))
```

```
##  [1] 1 0 0 1 0 0 0 0 1 1
```

```r
(w_excite=runif(n, min=0, max=1))
```

```
##  [1] 0.098731328 0.337059930 0.975776162 0.007023608 0.904275181
##  [6] 0.390917401 0.721229906 0.736430289 0.568050301 0.091370009
```
inhibitory inputs with values ${0, 1}$,

```r
m=8
(x_inhibit=sample(c(-1,0), m, replace=TRUE))
```

```
## [1] -1 -1  0  0 -1  0  0 -1
```

```r
(w_inhibit=runif(m, min=0, max=1))
```

```
## [1] 0.45798762 0.34654931 0.50930096 0.07108773 0.85400530 0.09431351
## [7] 0.64575739 0.66557760
```
and the threshold of $0$ as described above

```r
(u = 0)
```

```
## [1] 0
```
Calculate the weighted sum as shown above with bias of $0.5$.

```r
b = 0.5
(z = b + sum(x_excite, w_excite) + sum(x_inhibit, w_inhibit))
```

```
## [1] 8.975444
```
Calculate the neuron output of $y$ using the
[Heaviside function](https://en.wikipedia.org/wiki/Heaviside_step_function)

```r
(y = ifelse(z > u, 1.0, 0.0))
```

```
## [1] 1
```

# Perceptrons
The [Wikipedia](https://en.wikipedia.org/wiki/Perceptron) entry is a good place to start.
It turns out that the definition given in the previous section matches Wikipedia's
definition of a Perceptron.

We can rewrite our equation above as $y = \sigma(b + \sum_i x_i w_i)$ where $\sigma$
is called the transfer function.  In the case above, the transfer function is the
Heaviside function.

According to Wikipedia, "modern" perceptrons use functions like the
[sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) as the transfer 
function $\sigma$.  Examples of sigmoids include the logisitic function $\sigma(t) = \frac{1}{1+e^{-t}}$

```r
sigma <- function(t) 1/(1+exp(-t))

plot(x=seq(from=-6,to=6,length.out=100), y=sigma(seq(from=-6,to=6,length.out=100)), type="l", xlab="t", ylab=expression(paste(sigma,"(t)")) )
```

![](Neurons_files/figure-html/unnamed-chunk-6-1.png)<!-- -->
  
which has y-asymptotes at 0 and 1 and 
has the property that its derivative can be expressed as a function of itself,
$\sigma'(t) = \sigma(t) (1 - \sigma(t))$.

Another sigmoid functions used in as the activation function is the hyperbolic tangent
$\tanh(t) = \frac{1-e^{-2t}}{1+e^{-2t}} = 2 \sigma(2t)-1$ which has y-asymptotes -1 and 1.  Note that the derivative of $\tanh(t)$ is $1 - \tanh^2(t)$.

See LeCun's paper [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) for reasons for
choosing one or the other.
