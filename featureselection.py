import numpy as np
import sys

#normalize data from input matrix
def normalize(matrix):
	print("\nNormalizing the data...\n\n")

def main():
	print("Welcome to Bryce's Feature Selection Algorithm.\n")
	file = input('Type in the the name of the file to test:  ') 

	#read file into matrix:
	matrix = np.loadtxt(file)
	#normalize the data
	print(matrix)
	
if __name__ == "__main__": main()