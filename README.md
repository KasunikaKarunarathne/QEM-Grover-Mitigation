# Machine Learning-Based Quantum Error Mitigation for Grover's Algorithm

This repository contains the implementation, datasets, and experimental results for a novel Machine Learning-based Quantum Error Mitigation (ML-QEM) framework designed to improve the execution fidelity of Grover's Algorithm on noisy quantum hardware. The study compares two distinct physical backends: IBM Quantum cloud devices (Superconducting Qubits) and SpinQ Triangulum (NMR Qubits).

## Our Goal
Quantum computers are highly susceptible to noise, which degrades the performance of deep quantum circuits like those required for Grover's search algorithm. The primary goal of this project is to develop and validate a scalable Machine Learning-based Quantum Error Mitigation (ML-QEM) pipeline that can learn hardware-specific noise profiles and mitigate errors without the massive circuit overhead required by traditional QEM techniques (such as ZNE or PEC).

## What We Achieved
- **Two-Tower Neural Network Architecture**: Developed a custom Two-Tower model that fuses circuit features (depth, gate counts, topological structure) with hardware fingerprints (T1/T2 times, readout errors) to predict noise-free expectation values.
- **Cross-Platform Validation**: Successfully deployed and evaluated the ML-QEM pipeline on both **IBM Quantum (Superconducting)** and **SpinQ Triangulum (NMR)** platforms.
- **Significant Fidelity Improvements**: Demonstrated substantial reductions in error rates compared to unmitigated execution on physical hardware, specifically tailored for the highly structured oracle and diffusion operators in Grover's Algorithm.
- **Automated Dataset Generation**: Created a robust pipeline to generate diverse training datasets (`spinq_dataset_gen.py`) and extract hardware fingerprints dynamically.

## Project Structure (What's Included)

* **`notebooks/`**: Interactive Jupyter notebooks detailing the QEM pipelines and experimental workflows.
  * `1_Grover_Error_Mitigation.ipynb`: The core implementation of Grover's algorithm with preliminary noise analysis.
  * `2_ML_QEM_Pipeline_IBM.ipynb`: The complete ML-QEM pipeline optimized and trained for IBM Quantum superconducting devices.
  * `3_ML_QEM_Pipeline_NMR.ipynb`: The adapted ML-QEM pipeline for the SpinQ NMR platform, handling its unique decoherence characteristics.
* **`data/`**: Datasets, hardware fingerprints, and model weights collected during execution.
  * `ibm_cloud_data/`: PyTorch model weights (`.pt`) and NumPy arrays (`.npy`) containing Choi matrices and target values specific to the IBM Quantum environment.
  * `nmr_data/`: Experimental datasets (`spinq_dataset.csv`), normalized statistics, model weights (`.pth`), and inference results from SpinQ Triangulum.
  * `simulation_data/`: Base datasets, loss curves, error vs. depth plots, and other evaluation metrics.
* **`scripts/`**: Supporting Python scripts for automation and hardware interaction.
  * `spinq_dataset_gen.py` & `spinq_dataset_gen_extended.py`: Scripts for generating training data by executing randomized circuits and collecting hardware execution statistics.
  * `spiqit_simulation.py`: Utility for simulating circuits using the SpinQ environment.
  * `test_connection.py`: Basic script to test connection to the quantum backend.

## Code Availability
This repository is publicly released in conjunction with our research paper to ensure full reproducibility and transparency. All model architectures, dataset generators, and analysis notebooks are provided to allow independent verification of our findings.

## Getting Started
1. Clone the repository: `git clone https://github.com/KasunikaKarunarathne/QEM-Grover-Mitigation.git`
2. Install the required dependencies (Qiskit, PyTorch, etc.).
3. Navigate to the `notebooks/` directory and open the relevant pipeline notebook to reproduce the results.
