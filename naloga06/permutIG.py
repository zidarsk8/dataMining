from bisect import bisect_left as bisect
from random import shuffle
from math import log
import Orange
import numpy as np

# returns the number of bits set to 1 in a number
def countOnes(n):
	return bin(n).count("1")

# calculates information gain between two bitmaps (integers)
# x,y must be {n: number, c: numberOfOnes, s: sizeRepresented}
# I should get rid of the size since it's in x and y but too lazy
def gain(x,y):
	size = max(x["s"],y["s"])
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


# calculates and returns Info gain values for each pair of values from
# attributeArray and classArray
# the information is then pickled and if a pickle file exists it gets
# used insted of calculating again
def getOriginalGains(attribArr,classArr,clean=False):
	orig = {}
	for ci, clas in enumerate(classArr):
		orig[ci] = {} 
		for ai, attr in enumerate(attribArr):
			orig[ci][ai] = gain(clas,attr)
	return orig

# calculates info gain between the attrributeArray and random permutations of 
# classArray. 
# function returns a sorted list of all random gains for each pair of numbers
def getRandomGains(attribArr,clas,permutations):
	# najlazi je na zacetku hitr inicializirat randomGains tabelo
	rg = {}
	for a in range(len(attribArr)): rg[a] = []
	
	rArr = list(bin(clas["n"])[2:])
	for i in xrange(permutations): #for testing 
		shuffle(rArr)
		rc = {"n":int("".join(rArr),2), "s":clas["s"], "c":clas["c"]}
		[rg[ai].append(gain(rc,attr)) for ai, attr in enumerate(attribArr)]
	[a.sort() for a in rg.values()]
	return rg

def getGainValues(td, tl, orig, clas = 0, iterations = 100):
	#global orig, td, tl, X, y, m, n, maxLabel
	res = []
	rand = getRandomGains(td,tl[clas],iterations)
	# uzame random info gaine, in shrani tuple, lokacija (1 je najbolsa 0 najslabsa), stevilo nenicelnih, in index.
	# hkrati pa tudi filtrira da je stevilo nenicelnih vecje od filtra
	res.append(sorted([(bisect(rand[j],orig[clas][j])/float(iterations),td[j]["c"],j)\
			for j in xrange(len(orig[clas]))],reverse = True))
	return res


def binarizeXmean(x):
	xbin = int("".join(["1" if i>x.mean() else "0" for i in x]),2)
	return {"c": countOnes(xbin), "n":xbin,"s":len(x)}

def binarizeX(x,y):
	oldg = 0
	best = {}
	un = np.unique(x)
	if len(un) > 50:
		return binarizeXmean(x)
	meje = (un[1:]+un[:-1])/2
	for s in meje:
		xbin = int("".join(["1" if i>s else "0" for i in x]),2)
		xbin = {"c": countOnes(xbin), "n":xbin,"s":len(x)}
		g = gain(xbin, y)
		if g>oldg:
			best = xbin
			oldg = g
	return best
	#td = [int("".join(["1" if i>minval else "0" for i in x]),2) for x in X.T]
	#td = [{"c": countOnes(i), "n":i,"s":m} for i in td]
	
def getGains(X,y,permutations = 1000, nonzero=50):
	m = X.shape[0] 
	yy = int("".join([str(int(a)) for a in y]),2);
	tl = [{"c": countOnes(yy), "n":yy,"s":m}]
	td = [binarizeX(x, tl[0]) for x in X.T]
	orig = getOriginalGains(td,tl)
	gains = getGainValues(td,tl, orig, clas = 0, iterations = 1000)
	return (td,gains)



if __name__ == "__main__":
	
	print "reading data"
	data = Orange.data.Table("data/train.tab")
	
	X, y, _ = data.to_numpy()
	# m = rows, n = columns
	m,n = X.shape 
	
	yy = int("".join([str(int(a)) for a in y]),2);
	tl = [{"c": countOnes(yy), "n":yy,"s":m}]
	
	print "binarize"
	td = [binarizeX(x, tl[0]) for x in X.T]
	#td = [binarizeXmean(x) for x in X.T]
	#means = [x.mean() for x in X.T]
	#td = [int("".join(["1" if i>means[i] else "0" for i in x]),2) for i,x in enumerate(X.T)]
	#td = [{"c": countOnes(i), "n":i,"s":m} for i in td]
	
	print "calculating original gains"
	orig = getOriginalGains(td,tl)
	print "calculating random gains"
	gains = getGainValues(td,tl, orig, clas = 0, iterations = 1000)

	print "done"