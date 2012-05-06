from collections import defaultdict
from collections import Counter
from itertools import chain
from random import shuffle
from math import ceil
from time import time
from sets import Set
import cPickle
import Orange
import os

#dataSource = "data/"
dataSource = "minidata/"

def getTestData(clean=False):
	global dataSource
	pickleFile = dataSource+"testData.pickled"
	csvFile = dataSource+"testData.csv"
	if not clean and os.path.isfile(pickleFile):
		print "Reading: "+pickleFile
		return cPickle.load(open(pickleFile))
	else:
		t = time()
		print "Reading: "+csvFile
		f = open(csvFile)
		a = [[int(y) for y in x.strip().split("\t")] for x in f.readlines()]
		cPickle.dump(a,file(pickleFile,"w"),-1)
		print "Done: %.3fs" % (time()-t)
	return a

def getTrainingData(clean=False):
	global dataSource
	pickleFile = dataSource+"trainingData.pickled"
	csvFile = dataSource+"trainingData.csv"
	if not clean and os.path.isfile(pickleFile):
		print "Reading: "+pickleFile
		return cPickle.load(open(pickleFile))
	else:
		t = time()
		print "Reading: "+csvFile
		f = open(csvFile)
		a = [[int(y) for y in x.strip().split("\t")] for x in f.readlines()]
		cPickle.dump(a,file(pickleFile,"w"),-1)
		print "Done: %.3fs" % (time()-t)
	return a

def getTrainingLabels(clean=False):
	global dataSource
	pickleFile = dataSource+"trainingLabels.pickled"
	csvFile = dataSource+"trainingLabels.csv"
	if not clean and os.path.isfile(pickleFile):
		print "Reading: "+pickleFile
		return cPickle.load(open(pickleFile))
	else:
		t = time()
		print "Reading: "+csvFile
		tl = open(csvFile)
		arr = [[int(j) for j in i.strip().split(",")] for i in tl.readlines()]
		cPickle.dump(arr,file(pickleFile,"w"),-1)
		print "Done: %.3fs" % (time()-t)
	return arr

# returns a list of columnt indexes with less than n non-zero values
def getBadAtributes(arr,n):
	aa, rind = zip(*arr), []
	[rind.append(j) for j,i in enumerate(aa) if i.count(0) > len(i)-n]
	return rind

# remove columns listed in badAtributes from array
def filterColumns(arr, badAtributes):
	aa = zip(*arr)
	[aa.pop(i) for i in sorted(badAtributes,reverse=True) if len(aa)>i]
	return zip(*aa)

# output array as an integer csv file
def resultToCsv(arr,fn="", separator = " "):
	if fn == "":
		fn = "result_%d.csv" % time()
	f = file(fn,"w")
	f.write("\n".join([separator.join([str(x).replace("c","") for x in i]) for i in arr ]))
	f.flush()
	f.close()

def splitData(data, numOfSegments=10, segment = 0):
	stp = float(len(data))
	f = int(ceil(stp/numOfSegments)) * segment
	t = int(ceil(stp/numOfSegments)) * (segment+1)
	return data[f:t]

def splitLabels(l, numOfSegments=10, segment = 0 ):
	stp = float(max(sum(l,[])))
	f = int(ceil(stp/numOfSegments)) * segment
	t = int(ceil(stp/numOfSegments)) * (segment+1)
	return [[j for j in i if j>=f and j<t ] for i in l]



# make a training and test datasets from existing training data. 
def splitTrainingData(data, labels, numOfSegments=10, segment = 0):
	stp = len(labels)
	f = stp/numOfSegments * segment
	t = stp/numOfSegments * (segment+1)
	trainD = data[:f]+data[t:]
	trainL = labels[:f]+labels[t:]
	testD = data[f:t]
	testL = labels[f:t]
	return (trainD, trainL, testD, testL)

def precision(t,p):
	return float(len(Set(t).intersection(Set(p)))) / len(p)

def recall(t,p):
	return float(len(Set(t).intersection(Set(p)))) / len(t)

def fscore(t,p):
	per = precision(t,p)
	rec = recall(t,p)
	return 2.0 * (per*rec) / (per+rec) if per+rec > 0 else 0

def avgFscore(t,p):
	l = len(t)
	sum = 0
	for i in xrange(l):
		sum += fscore(t[i],p[i])
	return sum/l


def listToOrange(t,c):
	labelSet = set(sum(c, []))

	class_vars = [Orange.feature.Discrete("c%s" % i, values=["F","T"]) for i in labelSet]
	features = [Orange.feature.Continuous("%d" % i) for i in range(len(t[0]))]
	domain = Orange.data.Domain(features, False, class_vars=class_vars)

	data = Orange.data.Table(domain)
	for i in range(len(t)):
		d =	Orange.data.Instance(domain, list(t[i]))
		d.set_classes([["F", "T"][lab in c[i]] for lab in labelSet])
		data.append(d)
	return data

def listToOrangeSingleClass(X,y):
	features = [Orange.feature.Continuous("%d" % i) for i in range(len(X[0]))]
	class_var = Orange.feature.Discrete("class", values=["F","T"])
	domain = Orange.data.Domain(features + [class_var])

	data = Orange.data.Table(domain)
	for i in range(len(X)):
		d =	Orange.data.Instance(domain, list(X[i])+[["F", "T"][y[i]]])
		#d.set_class(["F", "T"][y[i]])
		data.append(d)
	return data

def tuplesToArr(pred,m=83):
	#res = [0 for i in range(max(max([max([i for i,j in p]) for p in pred])),m)+1]
	res = []
	for x,p in enumerate(pred):
		r = [0]*(m+1)
		for i,j in p:
			r[i] = j
		res.append(r)
	return res

def addBinValues(arr, threshold = 0):
	binD = [[int(x>0) for x in i] for i in arr]
	newD = []
	[newD.append(list(arr[i])+list(binD[i])) for i in range(len(arr))]
	return newD

# arr has to be [[(clas, prob),...]] with sorted by probability in 
# each test case
def normalizeTupple(arr):
	m = [x[0][1] for x in arr]
	return  [[(int(str(i).replace("c","")),j/m[x]) for i,j in y] for x,y in enumerate(arr)]

def probToRes(sortedRes,a=0.2, b=30.0):
	rr = []
	b = float(b)
	for r in sortedRes:
		rr.append([int(str(x[0]).replace("c","")) for i,x in enumerate(r) if x[1] > r[0][1]*(a+(i/b))])
	return rr

def addProb(a1,a2,w1,w2):
	ss = []
	for i in xrange(len(a1)):
		ss.append(defaultdict(float))
		for j in a1[i]:
			ss[i][j[0]] += j[1]*w1
		for j in a2[i]:
			ss[i][j[0]] += j[1]*w2
	return [sorted(s.items(),key=lambda x: x[1], reverse=True) for s in ss]

def probToFullArray(pred, numberOfLabels=83):
	res = []
	for p in pred:
		rres = []
		for x in p:
			new = [0 for a in range(numberOfLabels)]
			for a,b in x:
				new[a-1] = b
			rres.append( new)
		res.append(rres)
	return res

def filterGains(gains, minNonZero=20, top = 2000):
	return gains
