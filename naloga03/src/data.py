from collections import Counter
from itertools import chain
from random import shuffle
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

def removeLeastCommonData(oData, oLabels, least=5):
	data = oData[:]
	labels = oLabels[:]
	c = Counter(chain(*labels))
	lc = Counter.most_common(c)
	bb = sorted(list(Set([j for i,j in lc])))
	a = [x[0] for x in lc if x[1] < bb[5]]
	rem = [i for i,j in enumerate(labels) if len(Set(j).intersection(Set(a))) > 0 ]
	[labels.pop(x) for x in sorted(rem, reverse=True)]
	[data.pop(x) for x in sorted(rem, reverse=True)]
	return (data, labels)

def removeMostCommonData(oData, oLabels, count=20):
	data = oData[:]
	labels = oLabels[:]
	for iafsa in range(count):
		c = Counter(chain(*labels))
		lc = Counter.most_common(c)
		dlc = {}
		for l in lc: dlc[l[0]] = l[1]
		teze = [max([ dlc[y] for y in x])  for x in labels]
		teze = sorted([(y,x) for x,y in enumerate(teze)])
		rem = [x[1] for x in teze[-10:]]
		[labels.pop(x) for x in sorted(rem, reverse=True)]
		[data.pop(x) for x in sorted(rem, reverse=True)]
	return (data, labels)

# podvoji vrstice v katerih nastopajo atributi z zelo malo pojavitvami
def addFakeData(oData,oLabels,count=100,low=10):
	data = oData[:]
	labels = oLabels[:]
	for iafsa in range(count):
		c = Counter(chain(*labels))
		lc = Counter.most_common(c)
	
		dlc = {}
		for l in lc: dlc[l[0]] = l[1]
	
		#teze = [sum([ dlc[y]**2 for y in x])  for x in labels]
		teze = [sum([ dlc[y] for y in x])  for x in labels]
		teze = sorted([(y,x) for x,y in enumerate(teze)])
		tt = teze[:max(low*10,200)]
		shuffle(tt)
		duplicate = [x[1] for x in tt[:low]]
		dLabels = [labels[i][:] for i in duplicate]
		dData = [data[i][:] for i in duplicate]
		for ii in range(1):
			for i in range(len(duplicate)):
				labels.append(dLabels[i])
				data.append(dData[i])
	#shuflamo vrstice da niso vec lepo, pa poskrbimo da labele ostanejo 
	#pri svojem primeru
	sd = []
	[sd.append((data[i],labels[i])) for i in xrange(len(data))]
	shuffle(sd)
	ll = []
	dd = []
	for x,y in sd:
		dd.append(x)
		ll.append(y)
	return (dd, ll)

def resultToCsv(arr,fn=""):
	if fn == "":
		fn = "result_%d.csv" % time()
	f = file(fn,"w")
	f.write("\n".join([" ".join([str(x).replace("c","") for x in i]) for i in arr ]))
	f.flush()
	f.close()
	
