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

def plotDataPoints(a,b,lines):
	matplotlib.rcParams['axes.unicode_minus'] = False
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(a, b, 'o')
	for l in lines:
		ax.plot(l)
	ax.set_title("prikaz tock z prilegajoco premico")
	plt.show()

def plotContour(pl1,pl2):
	# Twice as wide as it is tall.
	fig = plt.figure(figsize=plt.figaspect(0.5))

	#---- First subplot
	# for 3d ax = fig.add_subplot(1, 2, 1, projection='3d')
	ax = fig.add_subplot(1, 2, 1)
	X = np.arange(-5, 5, 0.25)
	Y = np.arange(-5, 5, 0.25)
	X, Y = np.meshgrid(X, Y)
	R = np.sqrt(X**2 + Y**2)
	Z = np.sin(R)

	# for 3d
	# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
	#		        linewidth=0, antialiased=False)
	surf = ax.contour(X, Y, Z)

	fig.colorbar(surf, shrink=0.5, aspect=10)

	#---- Second subplot
	a,b,lines = pl2

	ax = fig.add_subplot(1,2,2)
	ax.plot(a, b, 'o')
	for l in lines:
		ax.plot(l)
	ax.set_title("prikaz tock z prilegajoco premico")


	plt.show()





data,result =  parseFile(dataset)

var = normalize(data[column])
val = normalize(result)

#plotDataPoints(normalize(data[0]),normalize(result),[[0,1],[0.5,0.1]])
pl2 = (normalize(data[0]),normalize(result),[[0,1],[0.5,0.1]])


plotContour(0,pl2)
