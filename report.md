Introduction
Deep neural networks are increasingly being used for modeling and simulating physical systems. The fusion of scientific computing and machine learning has given rise to PINNs, or physics-informed neural networks.
PINNs provide a model for complex forward and inverse problems involving PDEs, or partial differential equations. These models seamlessly blend noisy experimental data with physical laws in the learning process. The goal of this project is to use PINNs to learn the dynamics of tissue deformation, to begin simulating deformable tissue on simple geometries.

Through this study, we can enable PINNs to address an additional challenge to define an open area for research and further methodological advances.

The document is organized in the following sections:
1) PINNS, here we provide the formulation of the PINNs in the considered problem
2) Mathematical formulation, in this section we show the considerations and the mathematical steps for the membrane
3) NN implementation, where we explain the three types of neural networks we used, namely MLP, MLP with RWF and finally KAN
4) SetUp, here we show how the data and metrics are generated
5) Results, in this section we show the various trains carried out with the respective hyper parameters used and the results obtained
6) Conclusions and considerations, explanation of the results obtained and the deductions that we can draw from them