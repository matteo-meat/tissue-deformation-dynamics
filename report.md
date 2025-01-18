## Introduction
Deep neural networks are increasingly being used for modeling and simulating physical systems. The fusion of scientific computing and machine learning has given rise to PINNs, or physics-informed neural networks.
PINNs provide a model for complex forward and inverse problems involving PDEs, or partial differential equations. These models seamlessly blend noisy experimental data with physical laws in the learning process. The goal of this project is to use PINNs to learn the dynamics of tissue deformation, to begin simulating deformable tissue on simple geometries.

Through this study, we can enable PINNs to address an additional challenge to define an open area for research and further methodological advances.

The document is organized in the following sections:
1) PINNs, here we provide the formulation of the PINNs in the considered problem
2) Mathematical formulation, in this section we show the considerations and the mathematical steps for the membrane
3) NN implementation, where we explain the three types of neural networks we used, namely MLP, MLP with RWF and finally KAN
4) SetUp, here we show how the data and metrics are generated
5) Results, in this section we show the various trains carried out with the respective hyper parameters used and the results obtained
6) Conclusions and considerations, explanation of the results obtained and the deductions that we can draw from them

## PINNs
We consider PINNs in solving PDEs, partial differential equations, following the original formulation of Raissi et al.
Partial differential equations generally take the form:

2.1

where N[・] is a linear or nonlinear differential operator and u describes the unknown latent solution that is governed by the system of partial differential equations.

We also know that PDEs are subject to initial and boundary conditions:

2.2
2.3

where B[・] is a boundary operator corresponding to Dirichlet, Neumann, Robin or periodic boundary conditions.

The PDE residues can be defined as:

2.4

this is because u(t, x) can be represented by a deep neural network u_θ(t,x), where θ denotes all the tunable parameters of the network.

We can then say that it is possible to train a physics-based model by minimizing the following composite loss function:
L(θ) = L_{ic}(θ) + L_{bc}(θ) + L_{r}(θ)

To produce accurate and robust results we need to follow the following training pipeline mainly composed of three phases: non-sizing of PDEs, choosing suitable network architectures and employing appropriate training algorithms.

## NN implementation
This section introduces and explains the different types of neural networks used for training PINNs.
We need to carefully choose an appropriate network architecture, as this choice affects the success of physics-based neural networks.

### MLP
The first neural network used is the Multi-layer Perceptrons, MLP, which is used as a universal approximator to represent the latent functions of interest.

The latent functions take as input the coordinates of a space-time domain and predict the corresponding target solution.

A MLP is recursively defined by:
4.1
and has as final layer:
4.2
in all this we consider x belonging to R^d as input, g^{(0)}(x) = x, d_0 = d, the weight matrix in the l-th layer is defined as W^{(l)} belonging to R^{d_l × d_{l - 1}}, the activation function is sigma and theta = (W^{(1)},b^{(1)}, ..., W^{(L+1)}, b^{(L+1)}) represent the trainable parameters in the network.

We know that networks should be neither narrow and shallow, as they cannot capture complex nonlinear functions, nor broad and deep, as they can be difficult to optimize.

### MLP with RWF
The second type of network used is random weight factorization, RWF.

This type of network can continuously improve the performance of PINNs.

We know that MLPs are commonly trained by minimizing an appropriate loss function L(θ) via gradient descent.

To improve convergence, we factorize the weight parameters associated with each neuron in the network as follows:
RWF factors the weights associated with each neuron in the network as:
2.3
where w^{(k,l)} is a weight vector representing the k-th row of the weight matrix W^{(l)} belonging to R^{d_{l − 1}}, s^{(k,l)} is a trainable scaling factor assigned to each individual neuron belonging to R, and v^{(k,l)} belongs to R^{d_{l − 1}} .

Then we can write the proposed weight factorization as:
2.4 
with s belonging to R^{d_l}.

### KAN
We know that MLPs are inspired by the Universal Approximation Theorem, while in KANs we focus on the Kolmogorov-Arnold Representation Theorem.

With some modifications we allow KANs to outperform MLPs in terms of accuracy and interpretability.
KANs have fully connected structures, like MLPs, and we also know that MLPs have fixed activation functions on nodes, while KANs have learnable activation functions on edges and have no linear weight matrices at all. Each weight parameter is replaced by a univariate function parameterized as a spline. KAN nodes simply add the incoming signals without applying any nonlinearity.

The Kolmogorov-Arnold Representation Theorem states that if f is a multivariate continuous function on a bounded domain, then f can be written as a finite composition of continuous functions of a single variable and the binary operation of addition.

Considering f : [0, 1]^{n} -> R,
2.1
where phi_{q, p} : [0, 1] -> R and PHI_q : R -> R.
They then proved that the only true multivariate function
is addition.

These 1D functions can be non-smooth and even fractal, so in practice they may not be learned. For this behavior the Kolmogorov-Arnold representation theorem has essentially been sentenced to death in machine learning.

We can, however, deviate from the original equation and generalize the network to arbitrary widths and depths. We also know that most functions in science and everyday life are often smooth and have sparse compositional structures, which potentially facilitates smooth Kolmogorov-Arnold representations.

### Analysis: MLP with RWF
The second model whose training we will analyze is MLP with Random Weight Factorization.

We wanted to reuse the same learning rate and batch size parameters used for the MLP trainings, in order to be able to compare the results obtained. We also ran the trainings with the same activation functions tested in the first model.

For the trainings where we used learning_rate = 0.002203836177626117 and batchsize = 500 we obtained:

Avg train loss: 12151.300893020629; Avg val loss: 6810.801324252425 with the Tanh activation function, Avg train loss: 100505.3612638461; Average val loss: 6643.876216661061 with ReLU activation function and finally Average train loss: 93402.89770152498; Average val loss: 8460.147160775321 with SiLU activation function.

These are the best values ​​obtained for the average of the loss validation for each previous activation function: for the first one at epoch number 65 we have avg validation loss: 4225.283203125, for the second one at epoch number 251 we have avg validation loss: 557.2007446289062 and for the SiLU at epoch number 89 we have avg validation loss: 268.5632629394531.

For the trainings where we used learning_rate = 0.001 and batchsize = None we got:

Mean train loss: 125946.73594916044; Mean val loss: 38291.67793114506 with Tanh activation function, Mean train loss: 2724211.885513117; Mean val loss: 342154.02136381174 with ReLU activation function and finally Mean train loss: 569026.4964409722; Mean value loss: 27070.49837782118 with SiLU activation function.

Also in this second case we are going to observe the best values ​​obtained for the average of the validation loss for each activation function used: for the first one at epoch number 16 avg validation loss: 5147.599609375, for the ReLU at epoch number 30 avg validation loss: 23163.8359375 and for the last activation function we have at epoch number 39 avg validation loss: 5063.10498046875.