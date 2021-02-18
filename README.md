# Source code for GP4
This repo includes implementation of GP4 Algorithm describe in the paper *GP4: Gaussian Process Proactive Path Planning for the Stochastic On Time Arrival Problem*, and the benchmarks, namely [DOT](https://www.sciencedirect.com/science/article/pii/S0191261520303271), [PLM](https://ieeexplore.ieee.org/abstract/document/7273960?casa_token=rMAE3kIG0xkAAAAA:I6GYS4_RNCLbgSXtUE1kJg5e0opekcn9eFL9Z6HQli33LOEg6YpBjqJmeskW9nyDKT9oQN6MM-uV), and [OS-MIP](https://www.sciencedirect.com/science/article/pii/S0191261515301429) used for comparison.

## Dataset
The network data used for conducting simulations is available at [Networks.zip](https://drive.google.com/file/d/12L7PRDGWPF-S6sz-tFMFgfeBZWwQwVaV/view?usp=sharing). Please unzip the downloaded file into the same directory of the source code.
The dataset includes 8 networks/time slots. Each of them consists of two files:
- `.csv`, nodes and links of the network and mean of travel time of each link.
- `.npy`, covariance matrix used in experiments.

## Dependencies
- Python 3.6+
- NumPy
- SciPy
- Pandas
- NetworkX

## Description
The source code includes the following files:
- `main.py`, Sample codes for testing GP4 and benchmarks on simple network descriped in the paper (Subsection V-A). Parameters are also specified here.
- `GP4.py`, Implementation of GP4, Log-GP4 and Bi-GP4 Algorithms.
- `benchmark.py`, Our implementation of DOT, PLM and OS-MIP in GP-regulated stationary environment.
- `evaluation.py`, Functions that evaluate the performance of a path/DOT generated routing policy in terms of its posterior probability.
- `func.py`, Tool functions used through out the code.
