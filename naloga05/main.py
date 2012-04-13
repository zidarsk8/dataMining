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

# test za J
# X = np.array([[1,2,3,4],[2,2,2,2]])
# Y = np.array([5,6,7,8])
# omega = np.array([0,1,2])
# J(omega,X,Y) # more bit 0.0, ce je omega 1,1,2 more bit rezultat 2 ... itd
def J(omega,X,Y):
	# dodamo enke uspredi da normalno mnozimo z prvim theta
	X = np.append(np.ones((1,X.shape[1])),X,axis=0)
	h = X.T.dot(omega)
	return ((h-Y)**2).sum()/2


def getContour(X,Y,f,t,points):
	XX,YY = np.meshgrid(np.linspace(f,t,points),np.linspace(f,t,points))
	Z = np.zeros(XX.shape)
	for i in range(len(X)):
		Z += (XX + YY * X[i] - Y[i]) ** 2
	return (XX,YY,Z**(0.5))

def plotAllTheThings(pl1,pl2,thete):
	# Twice as wide as it is tall.
	fig = plt.figure(figsize=plt.figaspect(0.5))

	#---- First subplot
	ax = fig.add_subplot(1, 2, 1)
	X,Y,Z = pl1
	for t in thete:
		ax.plot(t[0],t[1],"x",color="blue")

	surf = ax.contour(X, Y, Z)
	fig.colorbar(surf, shrink=0.9, aspect=10)

	#---- Second subplot
	a,b = pl2
	ax = fig.add_subplot(1,2,2)
	ax.plot(a, b, 'o')
	for l in lines:
		ax.plot(l)
	ax.set_title("prikaz tock z prilegajoco premico")

	plt.show()




dataset = "data/x17.txt"
column = 1

data,result =  parseFile(dataset)

var = normalize(data[column])*5
val = normalize(result)

pl2 = (var,val)


pl1 = getContour(var,val,-10,10,100)

thete=[[1,2],[2,3],[2.5,5],[3,10]]

plotAllTheThings(pl1,pl2,thete)
