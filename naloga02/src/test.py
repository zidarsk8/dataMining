from random import random
from random import shuffle
from math import log


def listToInt(l,t,f):
	return int("".join(l).replace(t,"1").replace(f,"0"),2)

def listToInt(l):
	return int("".join(l).replace("T","1").replace("F","0"),2)

def countOnes(n):
	return len([a for a in list(bin(n)) if a=="1"])

def countOnesSlow(n):
	i = 1
	count = 0
	while i<=n:
		count += n&i != 0
		i *= 2
	return count

def entropy(n,size):
	p1 = countOnes(n)/float(size)
	p0 = 1-p1
	return -(p1 * log(p1,2) + p0 * log (p0,2))

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




aa = ["T"]*512+["F"]*(2000-512)
bb = ["T"]*512+["F"]*(2000-512)

shuffle(aa)
shuffle(bb)
a = listToInt(aa)
b = listToInt(bb)

print gain(8,10,4)
print gain(12,10,4)
print gain(10,10,4)
