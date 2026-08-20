import numpy as np
import torch
import random
from spinqit import get_basic_simulator, get_compiler, Circuit, BasicSimulatorConfig
from spinqit import H,CZ, X
#SpinQit defines 22 logic quantum gates 
# (I, H, X, Y, Z, Rx, Ry, Rz, P, T, Td, S, Sd, CX/CNOT, 
# CY, CZ, U, CP, SWAP, CCX, Ph, CSWAP) and 
# two special gates (MEASURE and BARRIER). 
# Specifically, P is the phase shift gate while Ph is the global phase gate.

# Setup the spinqit simulator 
compiler = get_compiler("basic_simulator")

num_samples =500
X_data = []
Y_data = []

def generate_spinq_grover_circuit(depth):
    circ = Circuit()
    q = circ.allocateQubits(2) # Allocate 2 qubits for NMR 

    available_gates = ['h0','h1','x0','x1','cz']
    cz_count =0
    for i in range(depth):
        gate = random.choice(available_gates)
        if gate == 'h0':
            circ << (H,q[0])
        elif gate == 'h1':
            circ << (H,q[1])
        elif gate == 'x0':
            circ << (X,q[0])
        elif gate =='x1':
            circ << (X,q[1])
        elif gate == 'cz':
            circ << (CZ,(q[0],q[1]))
            cz_count +=1

    return circ, cz_count

print(f"Generating {num_samples} training circuits on SpinQ Simulator...")

for i in range(num_samples):
    depth = random.randint(3,11)
    # 1 Buid the circuit and get the gate count 
    circ,cz_count = generate_spinq_grover_circuit(depth)

    # get the compiler and backend 
    comp = get_compiler("native")
    engine = get_basic_simulator()

    # Compile
    optimization_level = 0
    exe = comp.compile(circ,optimization_level)

    # Run 
    config = BasicSimulatorConfig()
    config.configure_shots(1024)
    result = engine.execute(exe,config)
    
    # 3 calculate the ZZ expectation value from pobabilities 
    # spinqit returns the probability distributions 
    # the zz expectation value = P(00) + P(11) - P(01) - P(10)
    probs = result.probabilities

    p_00 = probs.get('00',0.0)
    p_01 = probs.get('01',0.0)
    p_10 = probs.get('10',0.0)
    p_11 = probs.get('11',0.0)

    expectation_val = p_00 +p_11-p_01-p_10

    # Store the data [Depth,cz_count ,noisy_exp]
    ideal_exp = expectation_val
    noisy_exp = expectation_val # untill connect to real NMR

    X_data.append([depth,cz_count,noisy_exp])
    Y_data.append([ideal_exp])

    if (i+1)%100 == 0:
        print(f"[{i+1}/{num_samples}] SpinQ circuits executed...")

X_tensor = torch.tensor(X_data,dtype=torch.float32)
Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

print(f"\nFinal SpinQit Tensor Shape: {X_tensor.shape}")
torch.save(X_tensor, "SpinQ_X_features.pt")
torch.save(Y_tensor, "SpinQ_Y_targets.pt")
print("SpinQ Tensors saved! Ready for ML pipeline.")

