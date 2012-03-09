from time import time
from sets import Set
import pickle
import os

def getDataArray(clean=False):
	pickleFile = 'minidata/dataArray.pickled'
	if not clean and os.path.isfile(pickleFile):
		print "reading training data from pickle file"
		return pickle.load(open(pickleFile))
	else:
		t = time()
		print "Reading trainingData.csv"
		#f = open("minidata/trainingData.csv")
		f = open("minidata/trainingD.csv")
		a = [[int(y) for y in x.strip().split("\t")] for x in f.readlines()]
		arr = []
		for ii,i in enumerate(a):
			arr.append({})
			for c,j in enumerate(i):
				if j != 0:
					arr[ii][c] = j
		pickle.dump(arr,file(pickleFile,"w"),-1)
		print "Reading complete: %.3fs" % (time()-t)
	return arr

def getLabelsArray(clean=False):
	pickleFile = 'minidata/labelsArray.pickled'
	if not clean and os.path.isfile(pickleFile):
		print "reading labels from pickle file"
		return pickle.load(open(pickleFile))
	else:
		t = time()
		print "Reading trainingLabels.csv"
		print "Reading trainingLabels.csv"
		#tl = open("minidata/trainingLabels.csv")
		tl = open("minidata/trainingL.csv")
		arr = [[int(j) for j in i.strip().split(",")] for i in tl.readlines()]
		pickle.dump(arr,file(pickleFile,"w"),-1)
		print "Reading complete: %.3fs" % (time()-t)
	return arr

