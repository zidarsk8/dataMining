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
	maxD = 0
	for i in k:
		aa = a[i] if a.has_key(i) else 0
		bb = b[i] if b.has_key(i) else 0
		d += log(min(aa,bb)+1)
		maxD += log(max(aa,bb)+1)
	return float(d)/maxD



def getPredictionsRows(trainD,trainL,testD):
	def sk(x): return x[1]
	allLabels = list(Set(Chain(*trainL)))
	result = []
	abc = len(testD)
	train, trainStats = getDataDict(trainD)
	test, testStats = getDataDict(testD)
	for ab,primer in enumerate(test):
		sys.stdout.flush()
		sys.stdout.write("\r calculating classes %5d/%d " %(ab+1,abc))
		dists = {}
		maxDist = 0
		for i,j in enumerate(train):
			dists[i] = distDict(j,primer)
			maxDist = max(dists[i],maxDist)
		dd = sorted(dists.iteritems(), key=sk, reverse=True)
		res = {}
		for i in allLabels:
			res[i] = 0
		for i in range(100):
			for j in trainL[dd[i][0]]:
				#Integrate[(Cos[Pi*(x/y)] + 1)^2, x]
				#Plot[(Cos[Pi*(x/15)] + 1)^2/4, {x, 0, 15}]
				res[j] += (100.0-i)/100
				#res[j] += (float(dd[i][1])/maxDist)/100.0
		rs = sorted(res.iteritems(), key=sk, reverse=True)
		result.append(rs)
	sys.stdout.write("\r                                              \r")
	return result

def getKnnResults(trainD,trainL,testD,a=0.2,b=30):
	pred = getPredictionsRows(trainD,trainL,testD)
	rr = []
	for r in pred:
		rr.append([x[0] for i,x in enumerate(r) if x[1] > r[0][1]*(a+(i/b))])
	return rr


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


labels = data.getLabelsArray()
rawData = data.getDataArray()
stPrimerov = len(labels)



#bad = data.getBadAttributes(rawData,10)
#rawData = data.filterArr(rawData,bad)

k = 10;
print "starting %d fold cross validation" % k
print "number of cases: %d" % len(rawData)
print "number of attributes: %d" % len(rawData[0])

aaa = 0
allPred = []
for i in xrange(k):
	#sys.stdout.write("\r%2s/%2d done" % (i+1,k))
	f = stPrimerov/k * i
	t = stPrimerov/k * (i+1)
	trainD = rawData[:f]+rawData[t:]
	trainL = labels[:f]+labels[t:]
	testD = rawData[f:t]
	testL = labels[f:t]
	
	predictions = getKnnResults(trainD,trainL,testD,0.45,20)
	allPred += predictions

	avgf = avgFscore(testL,predictions)
	aaa += avgf
	print "%2d fscore : %.6f" % (i, avgf)

print "povpreceno (%f,%d) : %.6f" % (tol,mej,aaa/k)

#labels = data.getLabelsArray(True)
#rawData = data.getDataArray(True)
#stPrimerov = len(labels)
#
#testData = data.getTestArray(True)
#predictions = getPredictionsRows(rawData,labels,testData)
#f = file("result%d.csv" % time(),"w")
#f.write("\n".join([",".join([str(x) for x in i]) for i in predictions ]))
#f.flush()
#f.close()
#
