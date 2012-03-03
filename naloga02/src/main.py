from math import log
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


alfa = 0.05
shuffleCount = 500
mld=jrs.Data(discretized=True)

key = mld.classes.keys()[0]

key = "c40"
for feature in range(10):
	orangeGain = Orange.feature.scoring.InfoGain()

	data = mld.get_single_class_data(label=key)
	ig = orangeGain(data.domain.features[feature], data) 

	c = [d.get_class() for d in data]
	d = listToIntTF([str(a[data.domain.features[feature]])  for a in data],"> 0.000","<= 0.000")

	valList =  [a.value for a in c]
	#for i, dd in enumerate(bin(d)[2:]) :
	#	print dd, data[i+2000-len(bin(d)[2:])][data.domain.features[0]]
	print key
	numAttrib = d
	numClass = listToInt(valList)
	print len(valList)
	print gain(numClass,numAttrib,1999)
	print ig

#suma = np.zeros(len(data[0])-1)
#suma += np.greater(igp, ig)

