"""
AUTHOR: Luis Gabriel Q. del Rosario
DATE: 01 MAY, 2019
DESCRIPTION: Simulation of Deutsch-Jozsa algorithm using Linear Algebra
    - NOTE: FILENAME of input should be test_cases.txt
    - INPUT #1 is the number of qubits, INCLUDING the ancilla bit
    - INPUT #2 is the function mapping for f(x), separated by newlines
    - OUTPUT consists of the ff:
        - Probability distribution of final outcomes
        - Pie chart of the distribution
        - Verdict on wether the function is constant or balanced
"""
import gc
import matplotlib.pyplot as plt
import numpy as np
H = np.matrix(((1, 1), (1, -1))) / np.sqrt(2)
Z = np.matrix(((1, 0), (0, -1)))
X = np.matrix(((0, 1), (1, 0)))
P0 = np.matrix(((1, 0), (0, 0)))
P1 = np.matrix(((0, 0), (0, 1)))

def scale(qnum, op, num_qubits):
    """Generate a matrix that only applies gate 'op' to its respective qubit."""
    gate_list = [np.eye(2) for i in range(num_qubits)] # all qubits go through I
    gate_list[qnum] = op # this is the only qubit that has a gate that isn't I
    scaled = gate_list[0] # start scaling from the 1st qubit

    # get the kronecker product of all gates in gate_list
    for quantum_gate in gate_list[1:]:
        scaled = np.kron(scaled, quantum_gate)

    return scaled

def scale_all(op, num_qubits, skip_last=False):
    """Generate a matrix that applies 'op' to all qubits (skip_last means last qubit is I)."""
    scaled = op
    for repeat in range(num_qubits - (2 if skip_last else 1)):
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

def init_qubits(num_qubits):
    """Initialize all qubit vectors based on num_qubits."""
    q = [np.matrix([1,0]) for i in range(num_qubits)]
    return get_tensor(q)

def CNOT(control, target, num_qubits):
    """Generate CNOT based on the idea that CNOT = (P0 (x) I) + (P1 (x) X)."""
    term_0 = scale(control, P0, num_qubits)
    term_1 = (scale(control, P1, num_qubits) * scale(target, X, num_qubits))
    return term_0 + term_1

def run_algo(op_list):
    """Execute all operations in a given list (multiply all matrices to each other)."""
    ops = op_list[::-1] # reverse the list; after all, it's matrix mult.
    result = ops[0]
    for op in ops[1:]:
        result = result * op
    return result

def measure(result, num_qubits):
    """Omit the last qubit, combine probabilities of the same kind (e.g. 000/001, 100/101)"""
    measurement = np.zeros(2**(num_qubits-1))
    for index, value in enumerate(result.transpose().tolist()[0]):
        measurement[index >> 1] += value * value
    return measurement

def significant(n):
    """Check if value is significantly greater than 0."""
    return (n < -1e-10 or n > 1e-10)

def generate_pie_chart(measurement, num_qubits):
    """Generates a pie chart for the probability distribution of a given measurement."""
    x_labels = []
    measurement_to_plot = []

    # Only consider those that are significantly greater than 0.
    for r in range(2**(num_qubits-1)):
        if (significant(measurement[r])):
            x_labels.append(bin(r)[2:])
            measurement_to_plot.append(measurement[r])
    plt.pie(measurement_to_plot, labels=x_labels)
    plt.show()

def U(f_map, num_qubits):
    """Generate an oracle matrix based on the given function mapping."""
    # INSPIRED BY https://github.com/meownoid/quantum-python/blob/master/quantum.py

    U = np.zeros((2**num_qubits, 2**num_qubits)) # Start with a matrix of zeroes.
    
    # Quantum state looks like IN-IN-IN-IN-IN-IN-OUT
    for input_state in range(2**num_qubits): # For each possible input
        input_string = input_state >> 1 # remove OUT
        output_qubit = (input_state & 1) ^ (f_map[input_string]) # remove IN, XOR with f(IN)
        output_state = (input_string << 1) + output_qubit # the full state, with new OUT
        U[input_state, output_state] = 1 # set that part of U to 1
    return U

def print_probabilities(measurement, num_qubits):
    """Print the probability distribution of a measurement."""
    print ("\n\tPROBABILITY DISTRIBUTION OF OUTCOMES:\n")
    print ("\tOUTCOME\t\tP(n)")
    print ("\t-------\t\t----")
    for label, p in enumerate(measurement):
        print ("\t{0:0{1}b}\t\t{2:.2%}".format(label, num_qubits-1, p))

def deutsch_jozsa(f_map, num_qubits):
    """Run the Deutsch-Jozsa Algorithm. Returns T if constant and F if balanced."""
    op_list = [] # the list of operations

    op_list.append(init_qubits(num_qubits)) # Initialize qubits to |0>
    op_list.append(scale(num_qubits-1, X, num_qubits)) # Set last qubit to |1>

    # START: H on all qubits
    op_list.append(scale_all(H, num_qubits))

    # Apply oracle function based on user-input f_map
    op_list.append(U(f_map, num_qubits))

    # END: H on all but last qubit
    op_list.append(scale_all(H, num_qubits, skip_last=True))

    # RUN THE ALGORITHM
    result = run_algo(op_list)

    # Measure all but last qubit
    measurement = measure(result, num_qubits)

    # Finally, determine function type, and generate pie chart.
    # print_probabilities(measurement)
    # generate_pie_chart(measurement)
    
    # CONSTANT if measurement of |0> is positive, else BALANCED
    return True if significant(measurement[0]) else False

def main():
    gc.enable()
    test_cases_file = open("test_cases.txt", "r")

    for case_no, line in enumerate(test_cases_file):
        f_map = list(map(int, line.split()))
        num_qubits = f_map.pop(0)
        result = deutsch_jozsa(f_map, num_qubits)
        print("CASE {}: {}".format(case_no+1, ("CONSTANT" if result else "BALANCED")))
    test_cases_file.close()
main()