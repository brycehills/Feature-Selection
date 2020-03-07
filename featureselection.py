import numpy as np
import sys

#ref - https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
def normalize(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom
	
def crossoneout(matrix, classes, flist ,feat):
	dist = 0.0			# distance
	minDistance = 0.0	# current min distance
	nn = 0				# nearest neighbor
	correct = 0			# number of correctly classified instances				
	
	#iterate rows to test all instances
	for i in range(0,len(matrix)):
		for j in range(0,len(matrix)):
			if(j!=i):		#do not calc dist to self (otherwise dist will be zero)
				for(k in range(0,len(flist)):	#only calc with specified features
				#dist = euclidean()
				#if(dist < minDistance):
					#minDistance = dist
					#nn = j
		if(classes[i] == classes[nn]):
			correct+=1
			
	return correct/len(matrix)
				


def forwardSelection(matrix, features,classes):

	flist = [] 			# list of features
	best = []			# list of features with highest accuracy
	topscore = 0.0		# the current max accuracy score
	appendItem = 0.0	# current best item
	maxscore = 0.0		# the overall top acc 

	# find best
	for i in range(0, features.shape[1]):
		for j in range(0, features.shape[1]):
			if(j not in flist):
				#find score with current iteration
				#score = crossoneout(matrix, classes, flist, j) - have to create this in order to test
				print("Using feature(s): {", ','.join(flist), ", ",j, "}", "accuracy is: ", score*100, "%")
				if(score > topscore):
					topscore = score 
					appendItem = j
					
		flist.append(j)
		
		#make sure we save the actual best combination of feats
		if(topscore > maxscore):
			maxscore = topscore
			best = flist
	return best

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
	
	print("Normalizing the data...\n")
	features = normalize(features, np.min(features), np.max(features))
	
	printAlgorithms()
	choice = input("\nEnter Selection: ")
	
	if(choice == "1"):
		forwardSelection(matrix, features, classes) 
	
if __name__ == "__main__": main()