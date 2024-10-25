# Synthetic-Population-MCMC

Effects of Different Population Synthesis Models in the Epidemiological SEIR Agent-Based Model

This repository contains the code, data, and models used for the research article titled **"Effects of Different Population Synthesis Models in the Epidemiological SEIR (Susceptible-Exposed-Infected-Released) Agent-Based Model."**

## Overview

The study explores how errors introduced by different population synthesis methods affect the outcomes of epidemiological simulations. Specifically, the SEIR Agent-Based Model (ABM) is employed to simulate disease spread using synthetic populations generated through two popular techniques: Iterative Proportional Fitting (IPF) and Markov Chain Monte Carlo (MCMC).

## Key Findings

IPF-generated populations show higher error rates, especially in commuting patterns and spatial distributions, compared to MCMC-based populations.
Variations in population attributes, such as the working-class distribution, result in different infection peaks, with IPF predicting 42.2 new infections at peak and MCMC predicting 34.7.
The study emphasizes how minor inaccuracies in social network attributes can propagate through agent-based epidemiological models, leading to significant differences in disease spread predictions.

## Contents

Code: Implementation of IPF, MCMC, and SEIR-ABM models.

Data: Synthetic population datasets and validation metrics.

Documentation: Setup instructions and detailed analysis scripts.

This repository serves as a reference for researchers interested in synthetic population modeling and epidemiological simulations, highlighting the importance of model selection in generating accurate public health predictions.

## Contact
For questions or collaboration, please contact **muazimran27@gmail.com**.
