# Enemble of Single-Effect Neural Networks (ESNN)
This repo is the official implementation for "Enemble of Single-Effect Neural Network" (ESNN) framework. It contains example code for running ESNN on continuous and binary classification data.

## Installation and Dependencies
Python >= 3.7.4,
tensorflow >= 2.1.0,
tensorflow-probability >= 0.9.0,
keras >= 2.3.1,
matplotlib >= 3.1.2,
numpy >= 1.17.2,
Pillow >= 7.1.0,
scikit-learn >= 0.21.3,
scipy >= 1.4.1

## Code
1. Regression example: `ESNN_regression.py`
2. Classification example: `ESNN_binary.py`
3. Example code to generate simulated data and to also run the "Sum of Single-Effects" regression model (SuSiE) ([Wang et al. 2020](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12388)): `simu_example.R`

To run on your own data, one can simply change the file path in the code. The simulation file contains examples of how to generate case-control data in a genome-wide association (GWA) study under the liability threshold model and a toy regression example.

## RELEVANT CITATIONS

W. Cheng, S. Ramachandran, and L. Crawford. Uncertainty Quantification in Variable Selection for Genetic Fine-Mapping using Bayesian Neural Networks. _biorxiv_. 2022.02.23.481675.

## QUESTIONS AND FEEDBACK
For questions or concerns with the ESNN software, please contact [Wei Cheng](mailto:wei_cheng1@brown.edu) or [Lorin Crawford](mailto:lorin_crawford@brown.edu).

We welcome and appreciate any feedback you may have with our software and/or instructions.
