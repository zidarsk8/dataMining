from time import time
from sets import Set
import cPickle
import os

def getTestArray(clean=False):
	pickleFile = 'minidata/testArray.pickled'
	if not clean and os.path.isfile(pickleFile):
		print "reading training test from pickle file"
		return cPickle.load(open(pickleFile))
	else:
		t = time()
		print "Reading testData.csv"
		f = open("minidata/testData.csv")
		#f = open("minidata/trainingD.csv")
		a = [[int(y) for y in x.strip().split("\t")] for x in f.readlines()]
		cPickle.dump(a,file(pickleFile,"w"),-1)
		print "Reading complete: %.3fs" % (time()-t)
	return a

def getDataArray(clean=False):
	pickleFile = 'minidata/dataArray.pickled'
	if not clean and os.path.isfile(pickleFile):
		print "reading training data from pickle file"
		return cPickle.load(open(pickleFile))
	else:
		t = time()
		print "Reading trainingData.csv"
		f = open("minidata/trainingData.csv")
		#f = open("minidata/trainingD.csv")
		a = [[int(y) for y in x.strip().split("\t")] for x in f.readlines()]
		cPickle.dump(a,file(pickleFile,"w"),-1)
		print "Reading complete: %.3fs" % (time()-t)
	return a

def getLabelsArray(clean=False):
	pickleFile = 'minidata/labelsArray.pickled'
	if not clean and os.path.isfile(pickleFile):
		print "reading labels from pickle file"
		return cPickle.load(open(pickleFile))
	else:
		t = time()
		print "Reading trainingLabels.csv"
		tl = open("minidata/trainingLabels.csv")
		#tl = open("minidata/trainingL.csv")
		arr = [[int(j) for j in i.strip().split(",")] for i in tl.readlines()]
		cPickle.dump(arr,file(pickleFile,"w"),-1)
		print "Reading complete: %.3fs" % (time()-t)
	return arr

def getBadAttributes(arr,minl):
	aa, rind = zip(*arr), []
	[rind.append(j) for j,i in enumerate(aa) if i.count(0) > len(i)-minl]
	return rind

def filterArr(arr, badAttributes):
	aa = zip(*arr)
	[aa.pop(i) for i in sorted(badAttributes,reverse=True)]
	return zip(*aa)

