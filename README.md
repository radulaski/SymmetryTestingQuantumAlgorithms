# SymmetryTestingQuantumAlgorithms

This repo contains:
- A Qiskit implementation of an algorithm which uses the Hilbert-Schmidt distance to estimate the asymmetry of Lindbladian time evolution,
- Notebooks using that algorithm to estimate various asymmetries of the single-qubit amplitude damping channel, and an open two-qubit spin chain,
- A Mathematica notebook containing the analytical calculation of the asymmetry measure for the spin chain case.

## Usage

Install requirements:

`pip install -r requirements.txt`

The notebooks should execute correctly as written, though the spin chain simulation will take multiple hours to run.
