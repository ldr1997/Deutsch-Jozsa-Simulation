# Deutsch-Jozsa-Simulation
All the files in relation to Luigi's CS 199 thesis and Luigi, Brent and Cheska's CS 171 project will be put here. *TO-DO: Optimize using Numba or any other python CUDA libraries.*

# File Input
Name must be *test_cases.txt*. Each test case is separated by a newline, and each number by a space. The first number is the number of qubits, and the following numbers is a mapping of the given function (either 1 or 0).

# Running the Simulator
Just run it using python, and it should look for the file name in the same directory.

The main program is *deutsch_jozsa_simulator.py* which reads from a file. If you would like to manually input the function, run *deutsch_jozsa_simulator-user_input.py* instead.

# Analysis
Important documents for analysis can be found here:
- [Google Docs](https://docs.google.com/document/d/1rMLoWCHa7d1i_ChscfdBmQgr4zNYsaY8NyQ0loJWM2s/edit?usp=sharing)
- [Spreadsheet for Results](https://docs.google.com/spreadsheets/d/14cN_VMUVvN13PpC8_N8PmErT1YSEfK3dSdGNpcWrFto/edit?usp=sharing)
