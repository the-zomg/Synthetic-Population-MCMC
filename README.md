# Synthetic-Population-MCMC

This repository contains the code, data, and models used for the research article titled **"Effects of Different Population Synthesis Models in the Epidemiological SEIR (Susceptible-Exposed-Infected-Released) Agent-Based Model."**

## Overview

The study explores how errors introduced by different population synthesis methods affect the outcomes of epidemiological simulations. Specifically, the SEIR Agent-Based Model (ABM) is employed to simulate disease spread using synthetic populations generated through two popular techniques: Iterative Proportional Fitting (IPF) and Markov Chain Monte Carlo (MCMC).

## Key Findings

IPF-generated populations show higher error rates, especially in commuting patterns and spatial distributions, compared to MCMC-based populations.
Variations in population attributes, such as the working-class distribution, result in different infection peaks, with IPF predicting 42.2 new infections at peak and MCMC predicting 34.7.
The study emphasizes how minor inaccuracies in social network attributes can propagate through agent-based epidemiological models, leading to significant differences in disease spread predictions.

## Contents

Code: Implementation of MCMC for SEIR-ABM models.

Data: Synthetic population datasets and validation metrics.

Documentation: Setup instructions and detailed analysis scripts.

This repository serves as a reference for researchers interested in synthetic population modeling and epidemiological simulations, highlighting the importance of model selection in generating accurate public health predictions.

## Citation
If you use this code or dataset, please cite:
```
Imran, M. M. A., Kim, Y., Jung, G., & Kim, Y. B. (2023). EFFECTS OF DIFFERENT POPULATION SYNTHESIS MODELS IN THE EPIDEMIOLOGICAL SEIR (SUSCEPTIBLE-EXPOSED-INFECTEDRELEASED) AGENT-BASED MODEL. International Journal of Industrial Engineering, 30(3).
```

## Contact
For questions or collaboration, please contact **muazimran27@gmail.com**.
