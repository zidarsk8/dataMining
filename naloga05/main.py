# podatki za linearno regresijo so dobljeni na strani:
# http://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d.axes3d import Axes3D


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

def getContour(X,Y,f,t,points):
	XX,YY = np.meshgrid(np.linspace(f,t,points),np.linspace(f,t,points))
	Z = np.zeros(XX.shape)
	for i in range(len(X)):
		Z += (XX + YY * X[i] - Y[i]) ** 2
	
	return (XX,YY,Z**(0.5))

def plotAllTheThings(pl1,pl2):
	# Twice as wide as it is tall.
	fig = plt.figure(figsize=plt.figaspect(0.5))

	#---- First subplot
	ax = fig.add_subplot(1, 2, 1)
	X,Y,Z = pl1

	surf = ax.contour(X, Y, Z)
	fig.colorbar(surf, shrink=0.9, aspect=10)

	#---- Second subplot
	a,b,lines = pl2
	ax = fig.add_subplot(1,2,2)
	ax.plot(a, b, 'o')
	for l in lines:
		ax.plot(l)
	ax.set_title("prikaz tock z prilegajoco premico")

	plt.show()





data,result =  parseFile(dataset)

var = normalize(data[column])*2
val = normalize(result)

#plotDataPoints(normalize(data[0]),normalize(result),[[0,1],[0.5,0.1]])
pl2 = (var,val,[[0,1],[0.5,0.1]])


pl1 = getContour(var,val,-100,100,100)

print pl1

plotAllTheThings(pl1,pl2)
