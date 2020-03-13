import numpy as np
import math
import sys
# the small dataset, accuracy can be 0.89  when using only features 5 7 3; while on the large dataset, accuracy can be 0.949 when using only features 27 15 1.
#ref - https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
def normalize(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom
	
def euclidean(A, B, flist):
	sum = 0.0
	for k in range(0,len(flist)):
		delta = A[flist[k]]-B[flist[k]]
		sum += math.pow(delta,2)
	return math.sqrt(sum)

def crossoneout(matrix, classes, flist ,feat,forwards,k):
	dist = 0.0				# distance
	minDistance = 99999.0	# current min distance
	correct = 0				# number of correctly classified instances	
	neighborlist = []		# store nearest distances
	class1 = 0 
	class2 = 0

	if(forwards == 1):
		flist.append(feat)
	else:
		flist.remove(feat)
	#iterate rows to test all instances
	for i in range(0,len(matrix)):
		for j in range(0,len(matrix)):
			if(j!=i):		#do not calc dist to self (otherwise dist will be zero)
			
				dist = euclidean(matrix[i], matrix[j],flist)
		
				if(dist < minDistance):
					neighborlist.insert(0,j)
					minDistance = dist
					nearest = j
					
		# copy list of k nearest item
		knearest = neighborlist[0:k-1]

		# count classes of neighbors to determine classification
		for item in knearest:
			if classes[item] == 1:
				class1+=1
			else:
				class2+=1
		
		# determine majority classification
		if(class1>class2):
			majorityClass = 1
		else:
			majorityClass = 2
		
		# check if correct class and count++ if so
		if(classes[i] == majorityClass):
			correct+=1
			
		# reinitialize counters
		class1 = 0
		class2 = 0
		
					
					
		#if(classes[i] == classes[nearest]):
			#correct += 1 #inc num of success classifications
			
		minDistance = 99999.0
		
	if(forwards == 1):
		flist.remove(feat)
	else:
		flist.insert(0,feat)
			
	return float(correct)/len(classes)
			

def forwardSelection(matrix, features,classes,k):
	flist = [] 	# list of features
	best = []			# list of features with highest accuracy
	topscore = 0.0		# the current max accuracy score
	appendItem = 0.0	# current best item
	maxscore = 0.0		# the overall top acc 

	# find best - iterate cols
	for i in range(0, features.shape[1]):
		for j in range(0, features.shape[1]):
			if(j not in flist):
				#find score with current iteration
				score = crossoneout(features, classes, flist, j,1,k) 
				#print("score: ",score)
				print("Using feature(s): {", str(flist), ", ",j, "}", "accuracy is: ", score*100, "%")
				if(score > topscore):
					topscore = score
					appendItem = j
				
		#make sure we save the actual best combination of feats
		if(topscore > maxscore):
			maxscore = topscore
			best = flist.copy()
			best.append(appendItem)
			
		topscore = 0.0 #reset to find local maxima
		
		if(appendItem not in flist):			
			flist.append(appendItem)

			
	print("\n\nFinished search!! The best feature subset is ", str(best) ," which has an accuracy of ", maxscore*100 ,"%")
	return best
	
	
def backwardElimination(matrix, features,classes,k):
	flist = [] 			# list of features
	best = []			# list of features with highest accuracy
	topscore = 0.0		# the current max accuracy score
	appendItem = 0.0	# current best item
	maxscore = 0.0		# the overall top acc 
	
	# initialize flist
	for i in range(0, features.shape[1]):
		flist.append(i)

	# find best - iterate cols
	for i in range(0, features.shape[1]):
		for j in range(0, features.shape[1]):
			if(j in flist):
			
				#find score with current iteration
				score = crossoneout(features, classes, flist, j,2,k) 
				print("Removing feature(s): ", j , " from set ", str(flist), ", accuracy is: ", score*100, "%")
				if(score > topscore):
					topscore = score
					appendItem = j
					
		#make sure we save the actual best combination of feats
		if(topscore > maxscore):
			maxscore = topscore
			best = flist.copy()
			best.remove(appendItem)
			
		topscore = 0.0 #reset to find local maxima
		flist.remove(appendItem)
		
	print("\n\nFinished search!! The best feature subset is ", str(best) ," which has an accuracy of ", maxscore*100 ,"%")
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
	k = int(input('Select k for nearest neighbor:  '))
	#read file into matrix:
	matrix = np.loadtxt(fname, dtype = 'float')
	
	#save first column of classes
	classes = matrix[:,0]
	features = matrix[:,1:]
			
	print("This dataset has ", features.shape[1] ," features (not including the class attribute), with ",len(features)," instances. \n")
	
	print("Normalizing the data...\n")
	features = normalize(features, np.min(features), np.max(features))
	
	printAlgorithms()
	choice = input("\nEnter Selection: ")
	
	if(choice == "1"):
		forwardSelection(matrix, features, classes,k) 
		
	if(choice == "2"):
		backwardElimination(matrix, features, classes,k) 
	
if __name__ == "__main__": main()