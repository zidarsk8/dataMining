from math import log
import sys
import pickle
import Orange
import numpy
import jrs

def listToIntTF(l,t,f):
	return int("".join(l).replace(t,"1").replace(f,"0"),2)

def listToInt(l):
	return int("".join(l).replace("T","1").replace("F","0"),2)

def countOnes(n):
	return len([a for a in list(bin(n)) if a=="1"])

def gain(x,y,size):
	x1 = countOnes(x)
	y1 = countOnes(y)
	x1y1 = countOnes(x&y)
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
	return  px1y1 * log((px1y1)/(px1*py1) + (x1y1==0),2) + \
		px0y1 * log((px0y1)/(px0*py1) + (x0y1==0),2) + \
		px1y0 * log((px1y0)/(px1*py0) + (x1y0==0),2) + \
		px0y0 * log((px0y0)/(px0*py0) + (x0y0==0),2)

def getAttributTable():
	mldd = jrs.Data(discretized=True)
	mld = mldd.get_single_class_data(label=mldd.classes.keys()[0])
	attr = {}
	fl = len(mld.domain.features)
	for i in xrange(fl):
		sys.stdout.flush()
		sys.stdout.write("\rLoading Attribute Table: %3d%%" % (100*i/fl))
		attr[mld.domain.features[i].name] = listToIntTF([str(a[mld.domain.features[i]])  for a in mld],"> 0.000","<= 0.000")
	print "\rLoading Attribute Table: 100%%"
	return attr	
	#lol one liner, sam ni nic hitrejsi
	#[attr.append(listToIntTF([str(a[mld.domain.features[feature]])  for a in mld],"> 0.000","<= 0.000")) for i in xrange(len(mld.domain.features))]

def getClassTable():
	clas = {}
	mldRaw = pickle.load(open("minidata/trainingDataOD.pickled"))
	validLabels = [i.name for i in mldRaw.domain.class_vars]
	ll = len(validLabels)
	for x,i in enumerate(validLabels):
		sys.stdout.flush()
		sys.stdout.write("\rLoading Class Table: %3d%%" % (100*x/ll))
		clas[i] = listToInt([mldRaw[x][i].value for x in xrange(len(mldRaw))])
	print "\rLoading Class Table: 100%%"
	return clas


attribArr = getAttributTable()
classArr = getClassTable()

print "c40",classArr["c40"]
print "D_0",attribArr["D_0"]

