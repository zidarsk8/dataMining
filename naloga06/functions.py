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



def listToOrange(t,c):
	labelSet = set(sum(c, []))

	class_vars = [Orange.feature.Discrete("c%s" % i, values=["F","T"]) for i in labelSet]
	features = [Orange.feature.Continuous("%d" % i) for i in range(len(t[0]))]
	domain = Orange.data.Domain(features, False, class_vars=class_vars)

	data = Orange.data.Table(domain)
	for i in range(len(t)):
		d =	Orange.data.Instance(domain, list(t[i]))
		d.set_classes([["0", "1"][lab in c[i]] for lab in labelSet])
		data.append(d)
	return data

def listToOrangeSingleClass(t,c):
	features = [Orange.feature.Continuous("%d" % i) for i in range(len(t[0]))]
	class_var = Orange.feature.Discrete("class", values=["0","1"])
	domain = Orange.data.Domain(features + [class_var])

	data = Orange.data.Table(domain)
	for i in range(len(t)):
		d =	Orange.data.Instance(domain, list(t[i])+[["0", "1"][c[i]]])
		#d.set_class(["F", "T"][c[i]])
		data.append(d)
	return data

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
