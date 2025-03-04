from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister, AncillaRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import math
from qiskit.visualization import circuit_drawer

def quantum_addition(qc, a_bits, b_bits, k):
    """
    Perform quantum addition of two binary numbers represented by the qubits at a_bits and b_bits.
    The last two qubits in the circuit are used as ancilla bits for the calculation.
    
    Parameters:
        qc: QuantumCircuit object.
        a_bits: List of qubit locations for the first binary number.
        b_bits: List of qubit locations for the second binary number.
        k: Offset to shift circuit
    
    Returns:
        List of qubit locations where the output is stored.
    """

    

    # Number of qubits needed for this operation
    d = max(len(a_bits), len(b_bits)) * 2 + 1 + k  # We need extra qubits for ancilla and carry/overflow

    # Balance inputs
    balance_inputs(qc, a_bits, b_bits)
    
    # Perform the addition using qadd for each pair of bits
    for i in range(len(a_bits)):
        # qc = qadd(qc, 2 * (len(a_bits) - i - 1), d - 2)
        # qc.ancilla
        qc = qadd(qc, 2 * (len(a_bits) - i - 1), 0)
    
    # Balance outputs
    balance_outputs(qc, a_bits, b_bits)

    # movedata(qc, 3, 0, 3)
    # movedata(qc, 1, a, a+1)
    # movedata(qc, 3, 3, 0)
    
    # Measurement step
    qc.measure(range(d+1), range(d+1))
    
    # Return the qubit locations where the output is stored
    return list(range(0, len(a_bits) + 1))

def balance_inputs(qc, A, B):
    """
    Balance the input qubits by setting them to the correct states using X gates.
    """
    d = max(len(A), len(B))
    
    for i in range(d):
        if int(A[d - i - 1]):
            qc.x(d - i - 1)
    for i in range(d, 2 * d):
        if int(B[d - i - 1 + d]):
            qc.x(2 * d - i - 1 + d)
    
    qc.barrier()
    for i in range(d - 1):
        for j in range(d - i - 1):
            start_idx = j + 1 + 2 * i
            end_idx = i + d
            swapq(qc, start_idx, end_idx)
    
    return qc

def balance_add_inputs(qc, d, k):
    """
    Balance the input qubits by setting them to the correct states using X gates.
    """
    for i in range(d - 1):
        for j in range(d - i - 1):
            start_idx = j + 1 + 2 * i + k
            end_idx = i + d + k
            swapq(qc, start_idx, end_idx)
    
    return qc


def balance_outputs(qc, A, B):
    """
    Balance the output qubits by applying swap operations to the qubits in the correct order.
    """
    d = max(len(A), len(B)) - 1
    
    for i in range(d):
        for j in range(d - i):
            start_idx = j * 2 + 2 + i
            end_idx = start_idx + 1
            swapq(qc, start_idx, end_idx)
    
    qc.barrier()
    return qc

def balance_add_outputs(qc, d, k):
    """
    Balance the output qubits by applying swap operations to the qubits in the correct order.
    """
    d = d - 1
    
    for i in range(d):
        for j in range(d - i):
            start_idx = j * 2 + 2 + i + k
            end_idx = start_idx + 1 + k
            swapq(qc, start_idx, end_idx)
    
    # qc.barrier()
    return qc

def swapq(qc, A, B):
    """
    Swap two qubits using CNOT gates.
    """
    qc.cx(A, B)
    qc.cx(B, A)
    qc.cx(A, B)

def qadd(qc, k, a_n):
    """
    Add two binary numbers using quantum gates and return the updated circuit.
    This function assumes that k and a_n are indices of the qubits.
    """
    # qc.barrier()
    qc.cx(k + 2, qc.ancillas[0+a_n])
    qc.reset(qc.ancillas[1+a_n])
    qc.cx(qc.ancillas[0+a_n], k + 1)
    qc.cx(k + 1, qc.ancillas[1])
    qc.cx(k, qc.ancillas[1+a_n])
    qc.cx(qc.ancillas[0+a_n], k)
    qc.ccx(k, k + 1, qc.ancillas[0+a_n])
    qc.reset(k)
    qc.reset(k + 1)
    qc.reset(k + 2)
    qc.cx(qc.ancillas[1+a_n], k + 1)
    qc.cx(qc.ancillas[0+a_n], k)
    # qc.barrier()
    qc.reset(qc.ancillas[0+a_n])
    qc.reset(qc.ancillas[1+a_n])
    return qc

def movedata(qc, num, a, b):
    '''
        Moves num bits from index a to b
    '''
    if b > a: shr(qc, num, a, b)
    else: shl(qc, num, a, b)

def shr(qc, num, a, b):
    for i in range(0, num):
        # print('shr swap: ' + str(num-i+a-1) + ' <-> ' + str(num-i+b-1))
        swapq(qc, num-i+a-1, num-i+b-1)

def shl(qc, num, a, b):
    for i in range(0, num):
        # print('shl swap: ' + str(i+a) + ' <-> ' + str(i+b))
        swapq(qc, i+a, i+b)


def create_qaddition(qc, input_size, offset, anc_offset):
    # Takes in input at offset and uses (input_size + offset + 1) bits
    balance_add_inputs(qc, input_size, offset)

    # Adds the bits along input_size + offset + 1
    # With 1 for carry in
    for i in range(input_size):
        qc = qadd(qc, 2 * (input_size - i - 1) + offset, anc_offset)

    # Balance outputs
    balance_add_outputs(qc, input_size, offset)

    return [offset, input_size+1]



# Example usage
A = [0, 0, 0]  # Binary number A (4 bits)
B = [1, 1, 1]  # Binary number B (4 bits)

k = 0

# # Create a quantum circuit with enough qubits
num_qubits = 2 * max(len(A), len(B)) + 1 + 4

# # Create registers:
qr_main = QuantumRegister(num_qubits+13, name='q')     # Main qubits for your data
qr_anc  = AncillaRegister(4, name='a')  # Ancilla qubits for temporary operations
cr = ClassicalRegister(num_qubits+13, name='c')           # Classical bits for measurement

# # Build the circuit with both registers
qc = QuantumCircuit(qr_main, qr_anc, cr)

# Perform the quantum addition
# output_qbits = quantum_addition(qc, A, B, k)

# print(output_qbits)

'''

sz = 4
l1 = create_qaddition(qc, sz, 0, 0)
l2 = create_qaddition(qc, sz, sz*2+1, 2)
movedata(qc, l2[1], l2[0], l2[0]-l1[1]+1)
l3 = create_qaddition(qc, l1[1], l1[0], 2)

'''


# shr(qc, 3, 0, 2)
# shl(qc, 3, 2, 0)

# shl(qc, 2, 2, 0)

# movedata(qc, 3, 0, 2)
# movedata(qc, 3, 2, 0)
'''
1
0
1
0
0
1
0
1
0-

'''

qc.x()


sz = 4

l1 = create_qaddition(qc, sz, 0, 0)
l2 = create_qaddition(qc, sz, sz*2+1, 2)

print(l1)
print(l2)

movedata(qc, l2[1], l2[0], l1[1])
# # movedata(qc, 5, 9, 5)
# movedata(qc, l2[1], l2[0], l2[0]-l1[1]+1)
# # l3 = create_qaddition(qc, 5, 0, 2)
l3 = create_qaddition(qc, l1[1], l1[0], 2)
# print()
print(l3)
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

# Draw the circuit
compiled_circuit.draw('mpl', fold=40)
plt.show()

# circuit_img = circuit_drawer(qc, output='mpl', fold=80)

# # circuit_img.savefig('hadamard_gate.png') # save figure as PNG
# circuit_img.savefig('(4+4)+(4+4).svg') # save figure as SVG

# exit()

# Set up the simulator and run the circuit
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
result = simulator.run(compiled_circuit, shots=1).result()
counts = result.get_counts(compiled_circuit)

print("Measurement counts:")
print(counts)

# Draw the circuit
compiled_circuit.draw('mpl', fold=40)
plt.show()
