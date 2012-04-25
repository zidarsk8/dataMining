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
	data = zip(*[[float(y) for y in p.split(x.strip())][1 if removeIndex else 0:]\
			for x in lines[2+cols:] if x!=""])
	return (np.array(data[0:-1]).astype(float),np.array(data[-1]).astype(float))

def normalize(b):
	return (b-b.min())/(b.max()-b.min())

def normalizeColumns(bb):
	for i in range(len(bb)):
		bb[i] = normalize(bb[i])
	return bb

# test za J
# X = np.array([[1,2,3,4],[2,3,2,2]])
# Y = np.array([5,8,7,8])
# omega = np.array([0,1,2])
# J(omega,X,Y) # more bit 0.0, ce je omega 1,1,2 more bit rezultat 2 ... itd
def J(omega,X,Y):
	return (((X.T.dot(omega))-Y)**2).sum()/2

def getContour(X,Y,f,t,thete,points):
	#razpon = math.sqrt(((thete[0]-thete[-1])**2).sum())
	razpon = abs(thete[0]-thete[-1]).max() * 1.1
	XX,YY = np.meshgrid(\
			np.linspace(thete[-1][0]-razpon,thete[-1][0]+razpon,points),\
			np.linspace(thete[-1][1]-razpon,thete[-1][1]+razpon,points))
	Z = np.zeros(XX.shape)
	for i in range(len(X[1])):
		Z += (XX + YY * X[1][i] - Y[i]) ** 2
	return (XX,YY,Z**(0.5))

def plotAllTheThings(pl1,pl2,thete,lines, contourCount = 20,simbol="x",figname=""):
	# Twice as wide as it is tall.
	fig = plt.figure(figsize=plt.figaspect(0.4))

	#---- First subplot
	ax = fig.add_subplot(1, 2, 1)
	X,Y,Z = pl1
	for t in thete:
		ax.plot(t[0],t[1],simbol,color="blue")

	levels = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/contourCount)
	ax.contour(X, Y, Z,levels)
	ax.set_title("Konvergenca theta_0 in theta_1")


	#---- Second subplot
	a,b = pl2
	a = a[1]
	ax = fig.add_subplot(1,2,2)
	ax.plot(a, b, 'o')
	
	lsp = np.linspace(a.min(),a.max(),1+a.max()-a.min())
	print lsp
	for ll in lines:
		l = ll[0]+ll[1]*lsp
		ax.plot(lsp,l)
	ax.set_title("Prikaz tock z prilegajoco premico")
	if figname == "":
		plt.show()
	else:
		plt.savefig(figname)


def analiticna(X,y):
	X = X.T
	return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def stohasticna(X,y,start,alpha,eps,stepLimit=100,returnArray=False):
	jCount,m = X.shape
	newTh = np.ones(jCount)
	theta = newTh -10
	if len(start) == jCount :
		newTh = np.array(start).astype(float)
	thetas = [newTh]
	count = 0
	X = X.T
	while sum(abs(theta-newTh)) > eps and count < stepLimit:
		for i in range(m):
			theta = newTh
			newTh = np.array(theta)
			for j in range(jCount):
				newTh[j] = newTh[j] + alpha * (y[i] - newTh.dot(X[i])) * X[i][j]
			if returnArray and i%10 == 0: thetas.append(newTh)
		count +=1
	return (theta, thetas,count)

def batch(X,y,start,alpha,eps,stepLimit=100, returnArray=False):
	jCount = X.shape[0]
	newTh = np.ones(jCount)
	theta = newTh -10
	if len(start) == jCount :
		newTh = np.array(start).astype(float)
	thetas = [newTh]
	count = 0
	while sum(abs(theta-newTh)) > eps and count < stepLimit:
		theta = newTh
		h0 = X.T.dot(theta)
		newTh = np.array(theta)
		for j in range(jCount):
			newTh[j] = theta[j] + alpha * sum((y-h0)*X[j])
		if returnArray : thetas.append(newTh)
		count +=1
	return (theta,thetas , count)


if __name__ == "__main__":

	dataset = "data/x17.txt"
	column = 3
	plot = False

	data,result =  parseFile(dataset)

	if plot:
		# x28 , col 9, -0.5, *4 
		data = np.array([normalize(data[column])])-0.5
		data *= 5
		result = normalize(result)
		column = 0
	else:
		data = normalizeColumns(data)
		result = normalize(result)
		

	data = np.append(np.ones((1,data.shape[1])),data,axis=0)




	start = [2,0.5]

	anal = analiticna(data,result)
	print anal,J(anal,data,result)
	bt, batc, batcIter = batch(data,result,start,0.001,0.00005,10000)
	print bt,J(bt,data,result),batcIter
	st, stoh, stohIter = stohasticna(data,result,start,0.001,0.00001,10000)
	print st,J(st,data,result),stohIter


	#lines = []
	#lines.append(anal)
	#lines.append(batc[-1])
	#lines.append(stoh[-1])
	#pl1 = getContour(data,result,0,20,batc,100)
	#pl1 = getContour(data,result,0,20,stoh,100)
	#pl2 = (data,result)
	if plot:
		#plotAllTheThings(pl1,pl2,batc,lines,simbol=".")
		#plotAllTheThings(pl1,pl2,batc,lines,simbol=".",figname="batch_l_001.pdf")
		plotAllTheThings(pl1,pl2,stoh,lines,simbol=".",figname="stohasticna_l_01.pdf")


