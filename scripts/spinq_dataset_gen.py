import random
import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_experiments.library import ProcessTomography
from qiskit_experiments.framework import ExperimentData
from qiskit import transpile
import qiskit.qasm2 as qasm2
import tempfile
import os
from spinqit import get_compiler, Circuit
# Import cloud backend instead of local NMR
from spinqit import get_spinq_cloud, SpinQCloudConfig

# Initialize spinq cloud backend 
comp = get_compiler('qasm')

# Enter the cloud credentials 
# username = "Nethmini_Kar"
# keyfile = "C:/Users/nethm/.ssh/id_rsa"
username = "NethminiK"
keyfile = "D:/Academic UOP/Internship/NMR/id"

spinq_engine = get_spinq_cloud(username=username, keyfile=keyfile) 

import time
from spinqit.backend.spinq_cloud_backend import SpinQCloudBackend

def resilient_execute(self, ir, config):
    for i in range(10): # retry submission if it fails
        try:
            status, msg, task_code = self.submit_task(ir, config)
            break
        except Exception as e:
            if i == 9: raise e
            time.sleep(3)
            
    if status == 200 or status == 202:
        print(f'Task {task_code} has been submitted successfully. Polling resiliently...')
        while True:
            try:
                result = self.get_task_result(task_code)
                return result
            except Exception as e:
                err_str = str(e)
                if 'Read timed out' in err_str or '10060' in err_str or 'Timeout' in err_str or 'Connection aborted' in err_str or 'RemoteDisconnected' in err_str:
                    print(f'  [Network Timeout] Reconnecting to check Task {task_code}...')
                    time.sleep(3)
                else:
                    raise e
    else:
        raise Exception(f'Task submission failed: status code = {status}. Message = {msg}')

# Override the original execute method with our robust one
SpinQCloudBackend.execute = resilient_execute

config = SpinQCloudConfig()
config.configure_platform('gemini_vp') # Gemini 2 Qubit NMR
config.configure_shots(1024)
config.configure_task("dataset_gen", "Generating Dataset")

def compile_qiskit_to_spinq(qc, comp):
    # This bypasses the broken SpinQit 'qiskit' compiler by dumping to QASM first
    # SpinQ Cloud also does not support explicit measure gates (it measures automatically)
    qc_stripped = qc.copy()
    qc_stripped.data = [inst for inst in qc_stripped.data if inst.operation.name != 'measure']
    
    basis = ['id', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 'cx', 'cz']
    t_qc = transpile(qc_stripped, basis_gates=basis)
    fd, path = tempfile.mkstemp(suffix='.qasm')
    os.write(fd, qasm2.dumps(t_qc).encode('utf-8'))
    os.close(fd)
    exe = comp.compile(path, 0)
    os.remove(path)
    return exe

def get_choi_features_spinq(gate_circuit, physical_qubits, comp, engine, config):
    """
    Equivalent of `get_choi_features` using the SpinQit QASM Bridge.
    Extracts the physical noise footprint (Choi matrix) of a gate on SpinQ hardware.
    """
    # 1. Generate Qiskit Tomography Circuits
    qpt = ProcessTomography(gate_circuit, physical_qubits=physical_qubits)
    qiskit_circuits = qpt.circuits()
    
    # 2. Execute on SpinQit
    spinq_raw_data = []
    for qc in qiskit_circuits:
        exe = compile_qiskit_to_spinq(qc, comp)
        result = engine.execute(exe, config)
        spinq_raw_data.append(result.counts)
        
    # 3. Stitch counts back to Qiskit
    exp_data = ExperimentData(experiment=qpt)
    for i, circ in enumerate(qiskit_circuits):
        exp_data.add_data({"counts": spinq_raw_data[i], "metadata": circ.metadata})
        
    # 4. Run MLE Tomography and get Choi Matrix
    qpt.analysis.run(exp_data).block_for_results()
    flat_choi = np.array(exp_data.analysis_results("state").value).flatten()
    return np.concatenate((np.real(flat_choi), np.imag(flat_choi)))

# --- 1. GET HARDWARE FINGERPRINTS ---
qc_h, qc_x, qc_cz = QuantumCircuit(1), QuantumCircuit(1), QuantumCircuit(2)
qc_h.h(0)
qc_x.x(0)
qc_cz.cz(0, 1)

print("Extracting Hardware Fingerprints from SpinQit...")
hardware_fingerprint = np.concatenate([
    get_choi_features_spinq(qc_h, [0], comp, spinq_engine, config),
    get_choi_features_spinq(qc_x, [0], comp, spinq_engine, config),
    get_choi_features_spinq(qc_cz, [0, 1], comp, spinq_engine, config)
])
np.save("hardware_fingerprint.npy", hardware_fingerprint)
print("Hardware fingerprint saved to disk!")

# for daya 2 
# load tomography circuits 
# np.load("hardware_fingerprint.npy", hardware_fingerprint)
# # # --- 2. CIRCUIT GENERATION & NOISY EXPECTATION VALUES ---
# NUM_CIRCUITS = 500# Reduced for testing, set to 500 for actual dataset
# observable = SparsePauliOp(["ZZ"]) # Example observable

# def get_spinq_expectation(qiskit_circuit, observable, comp, engine, config):
#     """
#     Runs a Qiskit circuit on SpinQit and calculates the expectation value of the observable.
#     """
#     # Append measurements to all qubits to get counts
#     qc_measured = qiskit_circuit.copy()
#     qc_measured.measure_all()
    
#     # Run on SpinQit
#     exe = compile_qiskit_to_spinq(qc_measured, comp)
#     result = engine.execute(exe, config)
#     counts = result.counts
    
#     # Calculate expectation value (Example for ZZ)
#     total_shots = sum(counts.values())
#     expectation = 0
#     for bitstring, count in counts.items():
#         # Clean up bitstring format if SpinQ outputs tuples or spaces
#         bitstring = str(bitstring).replace(" ", "") 
        
#         # Parity calculation for ZZ (assuming 2 qubits)
#         parity = 1 if bitstring.count('1') % 2 == 0 else -1
#         expectation += parity * (count / total_shots)
        
#     return expectation

# print("Generating Dataset...")
# X_circuits_list, Y_targets_list = [], []

# def make_grover_11():
#     qc = QuantumCircuit(2, name="Grover_11")
#     qc.h([0, 1]); qc.cz(0, 1); qc.h([0, 1]); qc.x([0, 1]); qc.cz(0, 1); qc.x([0, 1]); qc.h([0, 1])
#     return qc

# def make_grover_00():
#     qc = QuantumCircuit(2, name="Grover_00")
#     qc.x([0, 1]); qc.h([0, 1]); qc.cz(0, 1); qc.h([0, 1]); qc.x([0, 1]); qc.cz(0, 1); qc.x([0, 1]); qc.h([0, 1]); qc.x([0, 1])
#     return qc

# def make_double_grover():
#     qc = QuantumCircuit(2, name="Double_Grover")
#     for _ in range(2):
#         qc.h([0, 1]); qc.cz(0, 1); qc.h([0, 1]); qc.x([0, 1]); qc.cz(0, 1); qc.x([0, 1]); qc.h([0, 1])
#     return qc

# def make_grover_with_rotations():
#     qc = QuantumCircuit(2, name="Grover_rotated")
#     qc.x(0); qc.h(1); qc.h([0, 1]); qc.cz(0, 1); qc.h([0, 1]); qc.x([0, 1]); qc.cz(0, 1); qc.x([0, 1]); qc.h([0, 1]); qc.h(0); qc.x(1)
#     return qc

# def generate_grover_basis_circuit(depth):
#     qc = QuantumCircuit(2)
#     available_gates = ["h0", "h1", "x0", "x1", "cz"]
#     for _ in range(depth):
#         gate = np.random.choice(available_gates)
#         if gate == "h0": qc.h(0)
#         elif gate == "h1": qc.h(1)
#         elif gate == "x0": qc.x(0)
#         elif gate == "x1": qc.x(1)
#         elif gate == "cz": qc.cz(0, 1)
#     return qc

# def extract_circuit_features(qc, noisy_exp):
#     ops = qc.count_ops()
#     f1 = qc.depth()                     # 0: Total Depth
#     f2 = ops.get("h", 0)                # 1: H-gate count
#     f3 = noisy_exp                      # 2: Raw Noisy Measurement
#     f4 = ops.get("cz", 0)               # 3: CZ-gate count
#     f5 = ops.get("x", 0)                # 4: X-gate count
#     f6 = f2 + f5                        # 5: Total 1Q gates
#     f7 = qc.num_nonlocal_gates()        # 6: Total 2Q gates
#     f8 = qc.width()                     # 7: Qubit width
#     return [f1, f2, f3, f4, f5, f6, f7, f8]

# from qiskit.primitives import StatevectorEstimator as IdealEstimator
# ideal_estimator = IdealEstimator()
# structured_funcs = [make_grover_11, make_grover_00, make_double_grover, make_grover_with_rotations]

# circuit_list, pub_list_ideal = [], []
# for _ in range(NUM_CIRCUITS):
#     qc = generate_grover_basis_circuit(random.randint(3, 11)) if random.random() < 0.8 else random.choice(structured_funcs)()
#     circuit_list.append(qc)
#     pub_list_ideal.append((qc, observable))

# print("Calculating ideal expectations using Qiskit StatevectorEstimator...")
# result_ideal = ideal_estimator.run(pub_list_ideal).result()

# for i in range(NUM_CIRCUITS):
#     qc = circuit_list[i]
#     ideal_exp = float(np.squeeze(result_ideal[i].data.evs))
    
#     # Get noisy expectation from SpinQ hardware
#     noisy_exp = get_spinq_expectation(qc, observable, comp, spinq_engine, config)
    
#     X_circuits_list.append(extract_circuit_features(qc, noisy_exp))
#     Y_targets_list.append([ideal_exp])

# X_circuits_t = torch.tensor(X_circuits_list, dtype=torch.float32)
# Y_targets_t = torch.tensor(Y_targets_list, dtype=torch.float32)
# print("--- SPINQ DATASET GENERATED SUCCESSFULLY ---")
