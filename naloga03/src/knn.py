#!/usr/bin/python2
import data
import sys
from math import log
from sets import Set
from time import time
from collections import Counter
from itertools import chain as Chain

def getDataDict(a):
	arr = []
	for ii,i in enumerate(a):
		arr.append({})
		for c,j in enumerate(i):
			if j != 0:
				arr[ii][c] = j
	sta = {}
	for i in xrange(len(a[0])):
		for j in xrange(len(a)):
			if a[j][i] != 0:
				if not sta.has_key(i):
					sta[i]=[0,0]
				sta[i][0] += a[j][i]
				sta[i][1] += 1
	return (arr,sta)

#def getDataStats(d,l):
#	stats = {}
#	for p in xrange(len(d)): #za vsak primer
#		for attr,val in d[p].items(): #gremo po atributih
#			for clas in l[p]: # in vseh razredih
#				if not stats.has_key(clas):
#					stats[clas] = {}
#				if not stats[clas].has_key(attr):
#					stats[clas][attr] = [0,0,0]
#				stats[clas][attr][0] += val
#				stats[clas][attr][1] += 1
#	return stats
#
#def getLinePrediction(line,stats,ds,ls):
#	result = {}
#	for i,j in stats.items():
#		result[i] = 0
#		for ii, jj in j.items():
#			print jj
#
#def getPredictionsEachAttr(trainD,trainL,testD):
#	result = []
#	d, ds = getDataDict(trainD)
#	l = list(trainL)
#	ls = Counter(Chain(*l))
#	stats = getDataStats(d,l)
#	for i in testD:
#		curRes = {}
#		for label, labelcount in ls.items():
#			curRes[label] = 0 # zacetna verjenost za razred i
#			for attr in testD[i]:
#				if testD[i][attr] != 0 and stats[label].has_key(attr):
#					curRes[label] += testD[i][attr]
#		result.append(curRes)
#	return result

#def dist(a,b):
#	l = min(len(a),len(b))
#	d = 0
#	maxD = 0
#	for i in xrange(l):
#		maxD += log(1+max(a[i],b[i]))
#		if min(a[i],b[i]) == 0:
#			d -= log(1+max(a[i],b[i]))
#		else:
#			d += log(1+min(a[i],b[i]))
#	return d #float(d)/maxD

def distDict(a,b):
	l = min(len(a),len(b))
	k = Set(a.keys()+b.keys())
	d = 0
	for i in k:
		if a.has_key(i) and b.has_key(i):
			d += min(a[i],b[i])
		#elif a.has_key(i):
		#	d -= 1#(1+a[i])
		#elif b.has_key(i):
		#	d -= 1#(1+b[i])
	return d #float(d)/maxD



def getPredictionsRows(trainD,trainL,testD):
	result = []
	abc = len(testD)
	train, trainStats = getDataDict(trainD)
	test, testStats = getDataDict(testD)
	for ab,primer in enumerate(test):
		sys.stdout.flush()
		sys.stdout.write("\r calculating classes %5d/%d " %(ab+1,abc))
		dists = {}
		for i,j in enumerate(train):
			dists[i] = distDict(j,primer)
		def sk(x): return x[1]
		dd = sorted(dists.iteritems(), key=sk, reverse=True)
		#TODO: eno bl pametno ibiranje razredov
		labs = []
		avgLabs = 0
		topK = 40
		topKi = 5
		for i in range(topKi):
			labs += trainL[dd[i][0]]
		for i in range(topK):
			labs += trainL[dd[i][0]]
			avgLabs += len(trainL[dd[i][0]])
		num = int((avgLabs/(topK)))
		result.append([x[0] for x in Counter.most_common(Counter(labs),num)])
		# dodamo num najbolj pogostih pojavljenih razredov ^
	sys.stdout.write("\r                                              \r")
	return result

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


labels = data.getLabelsArray(True)
rawData = data.getDataArray(True)
stPrimerov = len(labels)

#testData = data.getTestArray(True)
#predictions = getPredictionsRows(rawData,labels,testData)
#f = file("result%d.csv" % time(),"w")
#f.write("\n".join([",".join([str(x) for x in i]) for i in predictions ]))
#f.flush()
#f.close()

k = 10;
print "starting %d fold cross validation" % k
for i in xrange(k):
	#sys.stdout.write("\r%2s/%2d done" % (i+1,k))
	f = stPrimerov/k * i
	t = stPrimerov/k * (i+1)
	trainD = rawData[:f]+rawData[t:]
	trainL = labels[:f]+labels[t:]
	testD = rawData[f:t]
	testL = labels[f:t]
	
	predictions = getPredictionsRows(trainD,trainL,testD)
	
	print "%2d fscore : %.6f" % (i, avgFscore(testL,predictions))


labels = data.getLabelsArray(True)
rawData = data.getDataArray(True)
stPrimerov = len(labels)

testData = data.getTestArray(True)
predictions = getPredictionsRows(rawData,labels,testData)
f = file("result%d.csv" % time(),"w")
f.write("\n".join([",".join([str(x) for x in i]) for i in predictions ]))
f.flush()
f.close()

