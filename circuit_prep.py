import numpy as np
from typing import Dict, List
from qiskit.extensions import UnitaryGate
from qiskit import execute
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeLima


def hoeffding(epsilon, delta):
    """Generate the necessary number of shots by the Hoeffding inequality."""
    return int((2 / epsilon**2) * np.log(2 / delta)) + 1


def prepare_bell_state(circuit, qubit_a: int, qubit_b: int, k: int, l: int):
    """Prepare a 2-qubit Bell state involving `qubit_a` and `qubit_b` of `circuit`."""
    if (k, l) == (0, 0):
        circuit.h(qubit_a)
        circuit.cx(qubit_a, qubit_b)
    elif (k, l) == (0, 1):
        circuit.x(qubit_a)
        circuit.h(qubit_a)
        circuit.cx(qubit_a, qubit_b)
    elif (k, l) == (1, 0):
        circuit.h(qubit_a)
        circuit.x(qubit_b)
        circuit.cx(qubit_a, qubit_b)
    elif (k, l) == (1, 1):
        circuit.x(qubit_a)
        circuit.h(qubit_a)
        circuit.x(qubit_b)
        circuit.cx(qubit_a, qubit_b)
    else:
        raise ValueError


def amplitude_damping_channel(circuit, qubit_a: int, qubit_b: int, gamma: float):
    """
    Apply amplitude damping using a unitary operator and 'environment' qubit.
    Here `qubit_a` undergoes damping, and `qubit_b` is the 'environment'.
    """
    amplitude_damping_matrix = np.array(
        [
            [0, np.sqrt(gamma), -np.sqrt(1 - gamma), 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, np.sqrt(1 - gamma), np.sqrt(gamma), 0],
        ]
    )
    amplitude_damping_gate = UnitaryGate(
        amplitude_damping_matrix, label=r"$\mathcal{D}_{\gamma}$"
    )
    circuit.append(amplitude_damping_gate, [qubit_a, qubit_b])


def convert_to_bell_basis(circuit, qubit_a: int, qubit_b: int):
    """Change `qubit_a` and `qubit_b` to the Bell basis."""
    circuit.cx(qubit_a, qubit_b)
    circuit.h(qubit_a)


def run_circuit(ckt, shots, use_noise_model: bool = False):
    """
    Execute `ckt` on a simulator.
    If `use_noise_model` is True, we use the `FakeLima` noise model.
    """
    simulator = Aer.get_backend("qasm_simulator")
    if use_noise_model:
        device_backend = FakeLima()
        noise_model = NoiseModel.from_backend(device_backend)
        result = execute(
            ckt, backend=simulator, shots=shots, noise_model=noise_model
        ).result()

    else:
        result = execute(ckt, backend=simulator, shots=shots).result()

    return result.get_counts()


def compute_estimator(results: Dict[str, Dict[str, int]]):
    """
    Use the raw circuit results to compute the estimator of the trace,
    as described in the paper.
    """
    Y_t = 0
    shots = 0
    for k_l, sub_results in results.items():
        k_l = [int(char) for char in k_l]
        k_dot_l = np.dot(k_l[::2], k_l[1::2])

        for i_j, count in sub_results.items():
            i_j = [int(char) for char in i_j]
            i_dot_j = np.dot(i_j[::2], i_j[1::2])

            Y_t += count * ((-1) ** (k_dot_l + i_dot_j))
            shots += count

    Y = Y_t / shots
    return Y
