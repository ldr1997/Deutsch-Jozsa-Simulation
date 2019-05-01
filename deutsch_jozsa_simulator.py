"""
AUTHOR: Luis Gabriel Q. del Rosario
DATE: 01 MAY, 2019
DESCRIPTION: Simulation of Deutsch-Jozsa algorithm using Linear Algebra
    - INPUT #1 is the number of qubits, INCLUDING the ancilla bit
    - INPUT #2 is the function mapping for f(x), separated by newlines
    - OUTPUT consists of the ff:
        - Probability distribution of final outcomes
        - Pie chart of the distribution
        - Verdict on wether the function is constant or balanced
"""

import matplotlib.pyplot as plt
import numpy as np
import quantum_gates as gate

NUM_QUBITS = int(input("Enter number of qubits: "))
N = NUM_QUBITS - 1 # number of input bits
H = np.matrix(((1, 1), (1, -1))) / np.sqrt(2)
Z = np.matrix(((1, 0), (0, -1)))
X = np.matrix(((0, 1), (1, 0)))
P0 = np.matrix(((1, 0), (0, 0)))
P1 = np.matrix(((0, 0), (0, 1)))

def scale(qnum, op):
    """Generate a matrix that only applies gate 'op' to its respective qubit."""
    gate_list = [np.eye(2) for i in range(NUM_QUBITS)] # all qubits go through I
    gate_list[qnum] = op # this is the only qubit that has a gate that isn't I
    scaled = gate_list[0] # start scaling from the 1st qubit

    # get the kronecker product of all gates in gate_list
    for quantum_gate in gate_list[1:]:
        scaled = np.kron(scaled, quantum_gate)

    return scaled

def scale_all(op, skip_last=False):
    """Generate a matrix that applies 'op' to all qubits (skip_last means last qubit is I)."""
    scaled = op
    for repeat in range(NUM_QUBITS - (2 if skip_last else 1)):
        scaled = np.kron(scaled, op)
    if skip_last:
        scaled = np.kron(scaled, np.eye(2))
    return scaled

def get_tensor(vectors):
    """Compute the tensor product of all vectors in the given list."""
    tensor = vectors[0]
    for vector in vectors[1:]:
        tensor = np.kron(tensor, vector)
    return tensor.transpose()

def init_qubits():
    """Initialize all qubit vectors based on NUM_QUBITS."""
    q = [np.matrix([1,0]) for i in range(NUM_QUBITS)]
    return get_tensor(q)

def CNOT(control, target):
    """Generate CNOT based on the idea that CNOT = (P0 (x) I) + (P1 (x) X)."""
    term_0 = scale(control, gate.P0)
    term_1 = (scale(control, gate.P1) * scale(target, gate.X))
    return term_0 + term_1

def run_algo(op_list):
    """Execute all operations in a given list (multiply all matrices to each other)."""
    ops = op_list[::-1] # reverse the list; after all, it's matrix mult.
    result = ops[0]
    for op in ops[1:]:
        result = result * op
    return result

def measure(result):
    """Omit the last qubit, combine probabilities of the same kind (e.g. 000/001, 100/101)"""
    measurement = np.zeros(2**(NUM_QUBITS-1))
    for index, value in enumerate(result.transpose().tolist()[0]):
        measurement[index >> 1] += value * value
    return measurement

def significant(n):
    """Check if value is significantly greater than 0."""
    return (n < -1e-10 or n > 1e-10)

def generate_pie_chart(measurement):
    """Generates a pie chart for the probability distribution of a given measurement."""
    x_labels = []
    measurement_to_plot = []

    # Only consider those that are significantly greater than 0.
    for r in range(2**N):
        if (significant(measurement[r])):
            x_labels.append(bin(r)[2:])
            measurement_to_plot.append(measurement[r])
    plt.pie(measurement_to_plot, labels=x_labels)
    plt.show()

def U(f_map):
    """Generate an oracle matrix based on the given function mapping."""
    # INSPIRED BY https://github.com/meownoid/quantum-python/blob/master/quantum.py

    U = np.zeros((2**NUM_QUBITS, 2**NUM_QUBITS)) # Start with a matrix of zeroes.
    
    # Quantum state looks like IN-IN-IN-IN-IN-IN-OUT
    for input_state in range(2**NUM_QUBITS): # For each possible input
        input_string = input_state >> 1 # remove OUT
        output_qubit = (input_state & 1) ^ (f_map[input_string]) # remove IN, XOR with f(IN)
        output_state = (input_string << 1) + output_qubit # the full state, with new OUT
        U[input_state, output_state] = 1 # set that part of U to 1
    return U

def determine_type(measurement):
    """Determine whether a function is BALANCED or CONSTANT. Returns T if constant and F if balanced."""
    print ("\n\tPROBABILITY DISTRIBUTION OF OUTCOMES:\n")
    print ("\tOUTCOME\t\tP(n)")
    print ("\t-------\t\t----")
    for label, p in enumerate(measurement):
        print ("\t{0:0{1}b}\t\t{2:.2%}".format(label, N, p))
    # CONSTANT if measurement of |0> is positive, else BALANCED
    return True if significant(measurement[0]) else False

def deutsch_jozsa(f_map):
    """Run the Deutsch-Jozsa Algorithm."""
    op_list = [] # the list of operations

    op_list.append(init_qubits()) # Initialize qubits to |0>
    op_list.append(scale(N, gate.X)) # Set last qubit to |1>

    # START: H on all qubits
    op_list.append(scale_all(gate.H))

    # Apply oracle function based on user-input f_map
    op_list.append(U(f_map))

    # END: H on all but last qubit
    op_list.append(scale_all(gate.H, skip_last=True))

    # RUN THE ALGORITHM
    result = run_algo(op_list)

    # Measure all but last qubit
    measurement = measure(result)

    # Finally, determine function type, and generate pie chart.
    deutsch_jozsa_result = determine_type(measurement)
    print ("\nFINAL DECISION: " + ("CONSTANT" if deutsch_jozsa_result
                                              else "BALANCED"))
    generate_pie_chart(measurement)

def main():
    f_map = []
    print("Please enter the function mapping below, separated by space, for the following inputs:")
    for i in range(2**N):
        f_map.append(int(input("{:0{}b}: ".format(i, N))))
    deutsch_jozsa(f_map)

main()