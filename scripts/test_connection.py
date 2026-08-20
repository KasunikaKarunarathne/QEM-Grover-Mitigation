import sys
from spinqit import get_compiler, get_nmr, NMRConfig, Circuit
from spinqit.model.gates import H, CX

print("1. Initializing connection to Gemini Mini (10.21.139.30:8989)...")
comp = get_compiler('native')
spinq_engine = get_nmr() 
config = NMRConfig()
config.configure_shots(1024)
config.configure_ip('10.21.139.30')
config.configure_port(8989)
config.configure_account('admin', '2000')



print("2. Creating a native SpinQ Circuit...")
qc = Circuit()
q = qc.allocateQubits(2)
c = qc.allocateClbits(2)
qc << (H, [q[0]])
qc << (CX, [q[0], q[1]])

print("3. Compiling the circuit...")
exe = comp.compile(qc, 0) 

print("4. Sending payload over the network to the Quantum Computer...")
try:
    result = spinq_engine.execute(exe, config)
    print("\n--- CONNECTION SUCCESSFUL! ---")
    print("Received hardware counts:", result.counts)
except Exception as e:
    print("\n--- CONNECTION FAILED ---")
    print("Error:", e)
    sys.exit(1)
