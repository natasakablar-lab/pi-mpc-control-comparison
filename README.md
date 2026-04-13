# PI vs MPC Control under Constraints

This repository provides a simulation framework for comparing:

- PI control with anti-windup
- Model Predictive Control (MPC)

under input constraints.

## Features

- Closed-loop simulation
- Actuator saturation handling
- Multiple scenarios:
  - nominal operation
  - reference change
  - disturbance rejection

## Usage

Run the Streamlit application:

```bash
streamlit run app.py

## Description

The tool allows visualization and comparison of system responses, including:

output trajectories
control inputs
constraint handling
Associated paper

This implementation supports the results presented in the research paper:

"Comparative Analysis of PI Control with Anti-Windup and Model Predictive Control under Input Constraints", Kablar 2026.