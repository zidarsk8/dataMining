# podatki za linearno regresijo so dobljeni na strani:
# http://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


dataset = "data/x17.txt"

column = 3

# returns columns of data 
def parseFile(filename,removeIndex = True):
	import re
	f = open(filename,"r")
	lines = [x.strip() for x in f.readlines() if x[0] != "#" ]
	cols = int(lines[0].split(" ")[0])
	rows = int(lines[1].split(" ")[0])
	p = re.compile(r"[ ]+")
	data = zip(*[[float(y) for y in p.split(x.strip())][1 if removeIndex else 0:] for x in lines[2+cols:]])
	return (np.array(data[0:-1]).astype(float),np.array(data[-1]).astype(float))

def normalize(b):
	return (b-b.min())/(b.max()-b.min())

def normalizeColumns(bb):
	for i in range(len(bb)):
		bb[i] = normalize(bb[i])
	return bb

def plotDataPointsMulti(aa,b):
	matplotlib.rcParams['axes.unicode_minus'] = False
	fig = plt.figure()
	ax = fig.add_subplot(111)
	c = ["b","g","r","c"]
	for a in aa:
		ax.plot(a, b, 'o')
	ax.set_title('Using hypen instead of unicode minus')
	plt.show()

def plotDataPoints(a,b):
	matplotlib.rcParams['axes.unicode_minus'] = False
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(a, b, 'o')
	ax.set_title('Using hypen instead of unicode minus')
	plt.show()






data,result =  parseFile(dataset)

var = normalize(data[column])
val = normalize(result)

print data

plotDataPointsMulti(normalizeColumns(data),normalize(result))


