from os.path import isfile
from random import random
from random import shuffle
from math import log
from matplotlib import pyplot
import sys
import pickle
import Orange
import numpy
import jrs

def testOriginalGains(og,label,p=False):
	mld=jrs.Data(discretized=True)
	data = mld.get_single_class_data(label)
	l = len(og[label])
	s = 0
	for i in sorted(og[label].iterkeys()):
		ga = Orange.feature.scoring.InfoGain(i,data)
		if p:
			print "%10s        %.9f        %.9f        %.9f" % (i,og[label][i]-ga,og[label][i],ga)
		s += abs(og[label][i]-ga)
	print "Povprecna napaka za razred \"%s\":  %.9f" % (label,s/l)

def getOrangeRandomGains(rg,clas):
	mld = jrs.Data(discretized=True)
	data = mld.get_single_class_data(clas)
	orange = {}
	for attr in rg[clas]:
		a = [x.get_class().value for x in data]
		l = len(rg[clas][attr])
		orange[attr] = []
		for i in range(l):
			shuffle(a)
			[ex.set_class(a[i]) for i,ex in enumerate(data)]
			orange[attr].append(Orange.feature.scoring.InfoGain(attr,data))
	return orange

def testRandomGains(rg,clas,attr):
	mld=jrs.Data(discretized=True)
	data = mld.get_single_class_data(clas)
	l = len(rg[clas][attr])
	avgRg = sum(rg[clas][attr])/l
	avgOr = 0
	oRan = []
	for i in range(l):
		a = [x.get_class().value for x in data]
		shuffle(a)
		[ex.set_class(a[i]) for i,ex in enumerate(data)]
		
		o = Orange.feature.scoring.InfoGain(attr,data)
		oRan.append(o)
		avgOr += o
	pl(oRan,50,"test"+clas+attr+"Orange.pdf")
	pl(rg[clas][attr],50,"test"+clas+attr+"Random.pdf")
	print "%.9f     %.9f" % (avgRg,(avgOr/l))


def printRandomGainsForAttr(randomGains,originalGains,clas,attr):
	randomGains[clas][attr].sort
	for e,i in enumerate(randomGains[clas][attr]):
		if originalGains[clas][attr] <= i:
			print "%4d  - %.9f" % (e,i)
		else:
			print "%4d    %.9f" % (e,i)
	print "original : ", originalGains[clas][attr]

def pl(a,b,c):
	pyplot.hist(a,bins=b)
	pyplot.savefig(c)
	pyplot.close()


