# Transmission of Model: Agent-Based Model (ABM)

Agent-based model accompanying the paper **Language enables the cumulative cultural evolution of mental models** (A. Bluet et. al., 2026).
Archived version: https://doi.org/10.5281/zenodo.23973977

The model simulates the transmission of mental model studied in the paper using an agent-based model.

This repository is a companion to the main analysis repository: https://github.com/DrBlublu/Transmission-of-model-project.
Experimental data and analyses of the human participants are in that repository.

## Repository structure

## Requirements

- [Python 3.X] (see `environment.yml`)

## Installation

```bash
git clone https://github.com/DrBlublu/2023_Transmission_of_model_project_ABM.git
cd 2023_Transmission_of_model_project_ABM

conda env create -f environment.yml
conda activate ABM_gnome   # the name is on the first line of environment.yml

pip install -e .           # installs the ABM package defined in setup.py
```

## Usage

Run the simulations (this can take some time):

```bash
python scripts/run_ABM.py [--arguments]
```

Run the analysis using `scripts\analysis.rmd`\

## Data

All the data used to build the model can be found in the `data\` folder and comes from the main project: https://github.com/DrBlublu/Transmission-of-model-project

## License

The code in this repository is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 Alexandre Bluet.
