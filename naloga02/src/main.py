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
import test

# turns list of true and false strings into an int TTFT = 1101 = 13 
def listToIntTF(l,t,f):
	i = int("".join(l).replace(t,"1").replace(f,"0"),2)
	return {"n":i, "l":len(l), "c":countOnes(i)}

# turns a list of T and F to an integer representation
def listToInt(l):
	return listToIntTF(l,"T","F")

# returns the number of bits set to 1 in a number
def countOnes(n):
	return bin(n).count("1")

# calculates the entropy of a number, where the values of
# each bit represents one value
def entropy(n,size):
	p1 = countOnes(n)/float(size)
	p0 = 1-p1
	return -(p1 * log(p1,2) + p0 * log (p0,2))

# calculates information gain between two bitmaps (integers)
# x,y must be {n: number, c: numberOfOnes, s: sizeRepresented}
# I should get rid of the size since it's in x and y but too lazy
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
		
# turns a 2D array into a list of numbers, where each number presents
# the binary values in a column 
# the information is then pickled and if a pickle file exists it gets
# used insted of calculating again
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

# turns a 2D array into a list of numbers, where each number presents
# the binary values in a column 
# the information is then pickled and if a pickle file exists it gets
# used insted of calculating again
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

# calculates and returns Info gain values for each pair of values from
# attributeArray and classArray
# the information is then pickled and if a pickle file exists it gets
# used insted of calculating again
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

# calculates info gain between the attrributeArray and random permutations of 
# classArray. 
# function returns a sorted list of all random gains for each pair of numbers
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
		for i in xrange(permutations): #for testing 
		#for i in xrange(permutations if clas == "c40" else 10):
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

# filers all relevant artributes from according to the randomGainsArray 
# with the confidence of alpha
def getRelevantAttributes(orig, rand, alpha):
	relevant = {}
	print "Obtaining relevant attributes at Alpha = %.2f" % alpha
	for clas in orig:
		relevant[clas] = []
		for attr in orig[clas]:
			if rand[clas][attr][int((len(rand[clas][attr])-1)*(1-alpha))] <= orig[clas][attr]:
				relevant[clas].append(attr)
	return relevant

attribArr = getAttributTable()
classArr = getClassTable()
originalGains = getOriginalGains(attribArr,classArr)
randomGains = getRandomGains(attribArr,classArr,100)
relevant = getRelevantAttributes(originalGains, randomGains,0.05);

