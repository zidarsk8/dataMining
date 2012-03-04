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

def listToIntTF(l,t,f):
	i = int("".join(l).replace(t,"1").replace(f,"0"),2)
	return {"n":i, "l":len(l), "c":countOnes(i)}

def listToInt(l):
	return listToIntTF(l,"T","F")

def countOnes(n):
	return bin(n).count("1")

def entropy(n,size):
	p1 = countOnes(n)/float(size)
	p0 = 1-p1
	return -(p1 * log(p1,2) + p0 * log (p0,2))

def gain(x,y,size):
	x1 = x["c"]
	y1 = y["c"]
	x1y1 = countOnes(x["n"]&y["n"])
	x1y0 = x1 - x1y1
	x0y1 = y1 - x1y1
	x0y0 = size - x1y1 - x1y0 - x0y1
	px1 = x1 / float(size)
	py1 = y1 / float(size)
	px0 = 1 - px1
	py0 = 1 - py1
	px1y1 = x1y1 / float(size)
	px1y0 = x1y0 / float(size)
	px0y1 = x0y1 / float(size)
	px0y0 = x0y0 / float(size)
	return  (0 if x1y1==0 else px1y1 * log((px1y1)/(px1*py1),2)) + \
		(0 if x0y1==0 else px0y1 * log((px0y1)/(px0*py1),2)) + \
		(0 if x1y0==0 else px1y0 * log((px1y0)/(px1*py0),2)) + \
		(0 if x0y0==0 else px0y0 * log((px0y0)/(px0*py0),2)) 
		
def getAttributTable():
	attr = {}
	if isfile('minidata/attributeTable.pickled'):
		print "Loading Attribute Table from pickle file"
		attr = pickle.load(open("minidata/attributeTable.pickled"))
	else:
		mldd = jrs.Data(discretized=True)
		mld = mldd.get_single_class_data(label=mldd.classes.keys()[0])
		fl = len(mld.domain.features)
		for i in xrange(fl):
			sys.stdout.flush()
			sys.stdout.write("\rLoading Attribute Table: %3d%%" % (100*i/fl))
			attr[mld.domain.features[i].name] = listToIntTF([str(a[mld.domain.features[i]])  for a in mld],"> 0.000","<= 0.000")
		print "\rLoading Attribute Table: 100%"
		pickle.dump(attr,file("minidata/attributeTable.pickled","w"),-1)
	return attr	

def getClassTable():
	clas = {}
	if isfile('minidata/classTable.pickled'):
		print "Loading Class Table from pickle file"
		clas = pickle.load(open("minidata/classTable.pickled"))
	else:
		mldRaw = pickle.load(open("minidata/trainingDataOD.pickled"))
		validLabels = [i.name for i in mldRaw.domain.class_vars]
		ll = len(validLabels)
		for x,i in enumerate(validLabels):
			sys.stdout.flush()
			sys.stdout.write("\rLoading Class Table: %3d%%" % (100*x/ll))
			clas[i] = listToInt([mldRaw[x][i].value for x in xrange(len(mldRaw))])
		print "\rLoading Class Table: 100%"
		pickle.dump(clas,file("minidata/classTable.pickled","w"),-1)
	return clas

def getOriginalGains(attribArr,classArr):
	orig = {}
	if isfile('minidata/originalGains.pickled'):
		print "Loading Original Gains from pickle file"
		orig = pickle.load(open("minidata/originalGains.pickled"))
	else:
		cal = len(classArr)
		aal = len(attribArr)
		for ci, clas in enumerate(classArr):
			orig[clas] = {} 
			for ai, attr in enumerate(attribArr):
				if ai % 100 == 0:
					sys.stdout.flush()
					sys.stdout.write("\rCalculating original Gain values: %3d%%" % (100*(ci*aal+ai)/(aal*cal)))
				orig[clas][attr] = (gain(classArr[clas],attribArr[attr],2000))
		print "\rCalculating original Gain values: 100%"
		pickle.dump(orig,file("minidata/originalGains.pickled","w"),-1)
	return orig

def getRandomGains(attribArr,classArr,permutations):
	rg = {}
	cal = float(len(classArr))
	aal = len(attribArr)
	# najlazi je na zacetku hitr inicializirat randomGains tabelo
	for c in classArr:
		rg[c] = {} 
		for a in attribArr:
			rg[c][a] = []
	for ci, clas in enumerate(classArr):
		rArr = list(bin(classArr[clas]["n"])[2:])
		#for i in xrange(permutations): for testing 
		for i in xrange(permutations if clas == "c40" else 10):
			shuffle(rArr)
			rc = {"n":int("".join(rArr),2), "l":classArr[clas]["l"], "c":classArr[clas]["c"]}
			for ai, attr in enumerate(attribArr):
				if (ai%100 == 0):
					percent = (100*(aal*(ci*permutations+i)+ai)/(aal*cal*permutations))
					sys.stdout.flush()
					sys.stdout.write("\rCalculating random Gain values: %3.1f%%" % percent)
				rg[clas][attr].append(gain(rc,attribArr[attr],2000))
	print "\rCalculating random Gain values: 100.0%"
	print "Sorting calculated values"
	[x.sort() for iii, a in rg.items() for ii, x in a.items()]
	return rg

def getRelevantAttributes(orig, rand, alpha):
	relevant = {}
	print "Obtaining relevant attributes at Alpha = %.2f" % alpha
	for clas in orig:
		relevant[clas] = []
		for attr in orig[clas]:
			if rand[clas][attr][int((len(rand[clas][attr])-1)*(1-alpha))] < orig[clas][attr]:
				relevant[clas].append(attr)
	return relevant

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
	for i in randomGains[clas][attr]:
		if originalGains[clas][attr] < i:
			print " - %.9f" % i
		else:
			print "   %.9f" % i
	print "original : ", originalGains[clas][attr]

def pl(a,b,c):
	pyplot.hist(a,bins=b)
	pyplot.savefig(c)
	pyplot.close()


attribArr = getAttributTable()
classArr = getClassTable()
originalGains = getOriginalGains(attribArr,classArr)
randomGains = getRandomGains(attribArr,classArr,2000)
relevant = getRelevantAttributes(originalGains, randomGains,0.05);

testRandomGains(randomGains,"c40","D_251")
testRandomGains(randomGains,"c40","D_1404")
testRandomGains(randomGains,"c40","D_14")
testRandomGains(randomGains,"c40","D_141")
testRandomGains(randomGains,"c40","D_1")

#testOriginalGains(originalGains, "c40")
#for r in sorted(relevant.iterkeys()):
#	print r, len(relevant[r])
#
#ral = [len(relevant[r]) for r in relevant]
#pyplot.hist(ral,bins=83)
#pyplot.xlabel("st. nenicelnih atributov")
#pyplot.ylabel("st. primerov")
#pyplot.savefig("ral.pdf")
#pyplot.close()
