import numpy as np
import sys

#ref - https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
def normalize(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

#display algorithm UI
def printAlgorithms():
	print("\n\nSelect an Algorithm to use...\n")
	print("1. Forward Selection\n2. Backward Elimination\n3. Custom")

#main code
def main():
	#prompt for input file
	print("Welcome to Bryce's Feature Selection Algorithm.\n")
	fname = input('Type in the the name of the file to test:  ') 

	#read file into matrix:
	matrix = np.loadtxt(fname, dtype = 'float')
	
	#save first column of classes
	classes = matrix[:,0]	
	features = matrix[:,1:]
	
	print(features)
	
	features = normalize(features, np.min(features), np.max(features))
	
	print(features)
	
	printAlgorithms()
	input("\nEnter Selection: ")
	
if __name__ == "__main__": main()