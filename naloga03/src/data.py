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
		f = open("minidata/trainingData.csv")
		#f = open("minidata/trainingD.csv")
		a = [[int(y) for y in x.strip().split("\t")] for x in f.readlines()]
		#pickle.dump(a,file(pickleFile,"w"),-1)
		print "Reading complete: %.3fs" % (time()-t)
	return a

def getLabelsArray(clean=False):
	pickleFile = 'minidata/labelsArray.pickled'
	if not clean and os.path.isfile(pickleFile):
		print "reading labels from pickle file"
		return pickle.load(open(pickleFile))
	else:
		t = time()
		print "Reading trainingLabels.csv"
		tl = open("minidata/trainingLabels.csv")
		#tl = open("minidata/trainingL.csv")
		arr = [[int(j) for j in i.strip().split(",")] for i in tl.readlines()]
		pickle.dump(arr,file(pickleFile,"w"),-1)
		print "Reading complete: %.3fs" % (time()-t)
	return arr

