# README

## Overview

This repository contains code for running experiments in the paper: IRL Making a Difference “IRL”: Inverse Reinforcement Learning for Restless Multi-Armed Bandits with Applications in Maternal and Child Health. The code is organized into two Jupyter Notebook files: `WHIRL_risk_experiments.ipynb` and `WHIRL_state_experiments.ipynb`. These notebooks allow you to recreate the 2 main experiments presented in the paper. For the other experiments in the paper, you can easily modify this code to also test those out by modifying the `source` and `target` functions. We are unable to provide real world data with this paper due to it containing sensitive health information from India, but hope this open source code can facilitate future research ideas!

## Experiment Notebooks

1. `WHIRL_risk_experiments.ipynb`: This notebook corresponds to the experiments associated with Figure 4, 5, 6,and 7 in the paper. You can use this notebook to run experiments related to risk analysis. It does not include real-world data due to authorization limitations. However, you will find functions for generating synthetic data within the notebook.

2. `WHIRL_state_experiments.ipynb`: This notebook corresponds to the experiments associated with Figure 8 and 9 in the paper. It allows you to run experiments related to state analysis. Like the previous notebook, it does not include real-world data but provides functions for generating synthetic data.

## Getting Started

To get started with the experiments, follow these steps:

1. Clone or download this repository to your local machine.
2. Install requirements with `pip install -r requirements.txt`
3. You should be able to run the code with Python Version >= 3.8
4. Open the Jupyter Notebook environment on your machine with `jupyter notebook`

5. Open either `WHIRL_risk_experiments.ipynb` or `WHIRL_state_experiments.ipynb`, depending on the experiments you want to run (difference is explained above).

6. Follow the instructions and run the cells within the selected notebook to perform the experiments. The notebooks include code, comments, and documentation to guide you through the process.

7. If you need to generate synthetic data for the experiments, the notebook provides functions and instructions for doing so.


## Data Authorization

Please note that we are not authorized to include real-world data in this repository. You will need to provide your own data if you intend to use real-world data for the experiments. However, we have included functions in the notebooks for generating synthetic data to help you get started.

## Contact

Enjoy running the experiments, and we hope this code is helpful for your research or analysis! Please reach out if you have any questions.
