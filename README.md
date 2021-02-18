# Source code for GP4
This repo includes implementation of GP4 Algorithm describe in the paper *GP4: Gaussian Process Proactive Path Planning for the Stochastic On Time Arrival Problem.*, and the benchmarks used for comparison.

## Dataset
The network data used for conducting simulations is available at [Networks.zip](https://drive.google.com/file/d/12L7PRDGWPF-S6sz-tFMFgfeBZWwQwVaV/view?usp=sharing)
Please unzip the downloaded file into the same directory of the source code.
The dataset includes 8 networks/time slots. Each of them consists of two files:
- '.csv', nodes and links of the network and mean of travel time.
- '.npy', covariance matrix used in experiment.

## Dependencies
- Python 3.6+
- NumPy
- SciPy
- Pandas
- NetworkX

## Description
The source code includes the following files:
- 'main.py', Sample codes for testing GP4 and benchmarks on simple network descriped in the paper. Parameters are also specified here.
- 'GP4.py', Implementation of GP4, Log-GP4, Bi-GP4 Algorithm.
- 'benchmark.py', Our implementation of DOT, PLM, OS-MIP in GP-regulated stationary environment.
- 'evaluation.py', Functions that evaluate the performance of a path/DOT generated routing policy in terms of its posterior probability.
- 'func.py', Necessary tool functions used through out the code.
