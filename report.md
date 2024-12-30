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