## Simulation of Bacterial-Antibiotic Interactions Using Agent-based Modeling
> Abstract: Agent-based models are stochastic models constructed from the bottom up, meaning that
individual agents are assigned certain attributes. They are computer simulations used to study in-
teractions between people, things, places, and time (Agent-Based Modeling, 2022). In our context, we
used agent-based modeling to build a simulation model of bacterial growth and antibiotic resistance
to understand the complex interactions between bacteria and antibiotics with each other and their
environment. This approach offers a logical framework for deducing low-level biochemical details
about the individual molecular components to high-level pharmacodynamic parameters, like an
antibiotic's MIC (Minimum Inhibitory Concentration). However, note that this simulation of the
bacteria and antibiotics shown is for the general populations of bacteria and antibiotics in an envi-
ronment with predefined parameters and is only concerned with vertical transfer mechanisms of
antibiotic resistance genes. By applying the computational methods our work has developed, we
aim to give researchers a more efficient way to discover new ways to fight against antibiotic re-
sistance.

**Note this work has been completed as part of the course 02-251 Great Ideas in Computational Biology
at Carnegie Mellon University Qatar**

## Dependencies
This project manages dependencies using the conda system so please ensure that is [installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). You can also manually install the following libraries but there's not gurantee it would work:
- mesa
- numpy
- matplotlib
- seaborn
- importlib
- pandas

## Environment Setup
```
$ git clone https://github.com/tvlpirb/gcb-antibiotics-sim.git
$ cd gcb-antibiotics-sim
$ conda env create -f environment.yml
```

## Running
To run this program please see the relevant notebook in the root directory. If you're interested in 
reading the paper you can also find it in the root directory.
