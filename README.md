# Project 2 in TMA4320 - Introduction To Scientific Computations

This projects aims to model the heat dispersion in a 2D plane from heat sorces based on data from few sensors using physics informed neural networks. The code is divided in to three parts: the FDM, NN and PINN. The FDM uses a numerical approach to calculate the heat dispersion based on sensor measurements along all data points along the boundaries, and is used as the solution when training the PINN and NN. The NN is a standard neural network and use no physics boundary conditions in the loss function, while the PINN is based on the NN, with physics based boundary conditions in the loss function, and thus more input parameters as well. 

## Authors
- Kristoffer Eide
- Elisaveta Røste

Spring 2026

## Run the code
#### Install uv and set environment
First install uv via https://docs.astral.sh/uv/getting-started/installation/.
Then generate environment with
```bash
uv sync
```

#### Run scripts
Activate environment
```bash
# Mac/Linux
    source .venv/bin/activate
# Windows (PowerShell)
    .venv\Scripts\Activate.ps1
```
Then run 
```bash
python scripts/<path>
```

Our main code is in
- scripts/run_fdm.py
- run_sensordata.py
- scripts/run_nn.py
- scripts/run_pinn.py

run_fdm returns the time development of the spread of the heat for the numerical approach, while NN and PINN for the neural networks. run_sensordata also provides the heat development in each sensor as a standard plot. We have also created a smart fdm to adjust the heat source based on desired temperature, which can be ran by changing the function solve_heat_equation to smart_solve_heat_equation in run_fdm.

The project task, and our rapport is in the documents folder. 
