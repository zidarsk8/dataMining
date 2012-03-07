from time import time
from sets import Set

def getDataArray(b=False):
	t = time()
	print "Reading trainingData.csv"
	f = open("minidata/trainingData.csv")
	if b:
		arr = [[int(y=="0") for y in x.strip().split("\t")] for x in f.readlines()]
	else:
		arr = [[int(y) for y in x.strip().split("\t")] for x in f.readlines()]
	print "Reading complete: %.3fs" % (time()-t)
	return arr

def getLabelsArray():
	t = time()
	print "Reading trainingLabels.csv"
	tl = open("minidata/trainingLabels.csv")
	arr = [[int(j) for j in i.strip().split(",")] for i in tl.readlines()]
	m = max(map(max,arr))+1
	res = [[int(j in i) for j in range(m)] for i in arr]
	print "Reading complete: %.3fs" % (time()-t)
	return res

