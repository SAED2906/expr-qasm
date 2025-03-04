import ast
import json
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.circuit.library import MCXGate  # Multi-controlled X gate
from qiskit_aer import AerSimulator
from qiskit.qasm2 import dumps
import matplotlib.pyplot as plt
import qcfunction as qcf

def parse_expr(node):
    if isinstance(node, ast.BinOp):
        left = parse_expr(node.left)
        right = parse_expr(node.right)
        op = op_to_str(node.op)
        return [left, op, right]
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Expr):
        return parse_expr(node.value)
    elif isinstance(node, ast.UnaryOp):
        return [op_to_str(node.op), parse_expr(node.operand)]
    elif isinstance(node, ast.Paren):
        return parse_expr(node.operand)
    else:
        raise ValueError(f"Unsupported node type: {type(node)}")

def op_to_str(op):
    """Convert AST operator to string representation."""
    operators = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Mod: "%",
        ast.Pow: "**",
        ast.BitAnd: '&',
        ast.BitOr: '|',
        ast.BitXor: '^',
        ast.Not: '!'

    }
    return operators[type(op)]

def parse_equation(equation):
    tree = ast.parse(equation, mode='eval')
    return parse_expr(tree.body)

equation = "A + B + (C*D+(G-H))"
equation = "(A*A) % (P) + (C+H)*C + (H+24)"
equation = "(1 + 20 + (2*4+(1+5)))"
equation = "((1 + 20) + (2 + 4))"
# equation = "4 + 1 + (3*3)"
parsed = parse_equation(equation)
print(parsed)

print()

def solver(expr):
    print(expr)
    dta = list(expr)
    if dta[1] == '+': return int(dta[0]) + int(dta[2])
    if dta[1] == '*': return int(dta[0]) * int(dta[2])
    if dta[1] == '-': return int(dta[0]) - int(dta[2])
    if dta[1] == '/': return int(dta[0]) / int(dta[2])

def print_parsed(parsed, indent):
    for expr in parsed:
        if type(expr) == list:
            print_parsed(expr, indent + '\t')
        else:
            if type(expr) == int:
                print(indent + "{0:b}".format(int(expr)))
            else:
                print(indent + str(expr))

node = {
    'val1': '',
    'opp': '',
    'val2': ''
}
def noder(parsed):

    node = {
        'val1': '',
        'opp': '',
        'val2': ''
    }

    if type(parsed[0]) == list:
        val1 = noder(parsed[0])
        node["val1"] = val1
    else:
        node["val1"] = parsed[0]
    node["opp"] =   parsed[1]  
    if type(parsed[2]) == list:
        val2 = noder(parsed[2])
        node["val2"] = val2
    else:
        node["val2"] = parsed[2]


    return node

def calculate_qubits(node):
    total = 0

    v1_tot = 0
    v2_tot = 0

    val1 = node["val1"]
    # print(type(val1))

    if type(val1) == dict:
        v1_tot = calculate_qubits(val1)
    else:
        v1_tot = len("{0:b}".format(int(val1)))

    opp = node["opp"]

    val2 = node["val2"]
    # print(type(val2))
    if type(val2) == dict:
        v2_tot = calculate_qubits(val2)
    else:
        v2_tot = len("{0:b}".format(int(val2)))

    if opp == '+':
        total = max(v1_tot, v2_tot) + 1
    elif opp == '*':
        total = v2_tot + v1_tot
    elif opp == '-':
        total = max(v1_tot, v2_tot)

    if v1_tot + v2_tot > total:
        total = v1_tot + v2_tot
    

    return total


def temp(node):
    
    l1 = []
    l2 = []
    c = ['-']

    val1 = node['val1']
    val2 = node['val2']
    opp = node['opp']

    if type(val1) == int and type(val2) == int:
        if (len("{0:b}".format(int(val1))) > len("{0:b}".format(int(val2)))):
            val2 = val1
        else:
            val1 = val2
        


    if type(val1) == dict:
        l1 = temp(val1)
    else:
        l1 = [0]*len("{0:b}".format(int(val1)))

    
    if type(val2) == dict:
        l2 = temp(val2)
    else:
        l2 = [0]*len("{0:b}".format(int(val2)))


    if opp == '+':
        # if len(l1) > len(l2):
        #     l2 = l1
        # else:
        #     l1 = l2
        c = (l1 + ['+'] + l2)
    elif opp == '*':
        c = (l1 + ['*'] + l2)


    return c

    


print(calculate_qubits(noder(parsed)))
readable_json = json.dumps(noder(parsed), indent=4)
print(readable_json)

print(temp(noder(parsed)))
# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
# from qiskit_aer import AerSimulator


# n = 3
# m = 4
# k = 1
# r = max(n, m)  # r = 4
# num_qubits_main = k + 2 * r  # For k=1 and r=4, we need at least 9 main qubits

# # Create the main quantum register and two classical registers:
# qr_main = QuantumRegister(num_qubits_main, 'main')
# # One classical register for the sum (r bits) and one for the carry (r+1 bits)
# cr_sum = ClassicalRegister(r, 'c_sum')
# cr_carry = ClassicalRegister(r+1, 'c_carry')
# qc = QuantumCircuit(qr_main, cr_sum, cr_carry)

# # --- Initialization ---
# # To correctly pad the numbers, initialize each number using r bits.
# first_value = 3   # Expected: 3
# second_value = 5  # Expected: 5

# # Use r-bit binary strings
# first_bin = bin(first_value)[2:].zfill(r)   # e.g. "0011" for 3
# second_bin = bin(second_value)[2:].zfill(r)   # e.g. "0101" for 5

# # Initialize first number in qubits [k, k+r-1]:
# for i, bit in enumerate(reversed(first_bin)):
#     if bit == '1':
#         qc.x(qr_main[k + i])
        
# # Initialize second number in qubits [k+r, k+2*r-1]:
# for i, bit in enumerate(reversed(second_bin)):
#     if bit == '1':
#         qc.x(qr_main[k + r + i])

# # --- Append the quantum adder ---
# qcf.quantum_adder(qc, n, m, k)

# # --- Measurement ---
# # Measure the sum register (which is the second number's qubits)
# for i in range(r):
#     qc.measure(qr_main[k + r + i], cr_sum[i])
# # Measure the carry register (the final qubit c_qubits[r] is the overflow)
# carry_reg = qc.qregs[-1]  # The last added register is the carry register
# for i in range(r+1):
#     qc.measure(carry_reg[i], cr_carry[i])

# # --- Simulation ---
# simulator = AerSimulator()
# compiled_circuit = transpile(qc, simulator)
# result = simulator.run(compiled_circuit, shots=1024).result()
# counts = result.get_counts(compiled_circuit)
# print("Measurement counts:")
# print(counts)


# qc.draw('mpl')
# plt.show()

# qc_example.measure_all()
# simulator = AerSimulator()

# # Compile the circuit for the simulator
# compiled_circuit = transpile(qc_example, simulator)

# # Execute the quantum circuit on the simulator
# result = simulator.run(qc_example).result()

# # Print the measurement results
# print(result.get_counts(qc_example))


# # Print the readable JSON


# print_parsed(parsed, '')

# print("{0:b}".format(int(10)))

# print("{0:b}".format(int(225)))
# print(len("{0:b}".format(int(225))))

# print(noder(parsed))