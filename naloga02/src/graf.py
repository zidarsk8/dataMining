from random import random
from math import log
from matplotlib import pyplot
import numpy 

def multiClassAttr(arr,legend,append="",title='Stevilo pomembnih znacilk za razred'):
	plt = []
	for i in arr:
		plt.append([len(i["c"+str(y)]) for y in  sorted([int(x[1:]) for x in i])])
	#plt2 = [len(arr[1]["c"+str(y)]) for y in  sorted([int(x[1:]) for x in arr[1]])]
	ind = numpy.arange(len(plt[0])) 
	width = 1
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	rects = []
	colors = ["b","r","g","y"]
	for i,p in enumerate(plt):
		rects.append(ax.bar(ind, p, width, color=colors[i%len(colors)]))
	#rects2 = ax.bar(ind, plt[1], width, color='r')
	ax.set_ylabel('Stevilo znacilk')
	ax.set_title('Scores by group and gender')
	ax.set_xlabel("Razred")
	tup = tuple([r[0] for r in rects])
	ax.legend( tup, legend )
	pyplot.savefig("../mu"+append+".pdf")
	pyplot.close()

def doubleClassAttr(arr1,arr2,append="",title='Stevilo pomembnih znacilk za razred'):
	plt1 = [len(arr1["c"+str(y)]) for y in  sorted([int(x[1:]) for x in arr1])]
	plt2 = [len(arr2["c"+str(y)]) for y in  sorted([int(x[1:]) for x in arr2])]
	ind = numpy.arange(len(plt1)) 
	width = 1
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	rects1 = ax.bar(ind, plt1, width, color='b')
	rects2 = ax.bar(ind, plt2, width, color='r')
	ax.set_ylabel('Stevilo znacilk')
	ax.set_title('Scores by group and gender')
	ax.set_xlabel("Razred")
	ax.legend( ([rects1[0], rects2[0]]), ('<', '<=') )
	pyplot.savefig("../db"+append+".pdf")
	pyplot.close()

def singleClassAttr(arr,append="",title='Stevilo pomembnih znacilk za razred'):
	plt = [len(arr["c"+str(y)]) for y in  sorted([int(x[1:]) for x in arr])]
	ind = numpy.arange(len(plt)) 
	width = 1
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	rects1 = ax.bar(ind, plt, width, color='b')
	ax.set_ylabel('Stevilo znacilk')
	ax.set_title(title)
	ax.set_xlabel("Razred")
	pyplot.savefig("../sc"+append+".pdf")
	pyplot.close()

##testing
#l = ('<', '<=', "==")
#a1 = {"c0":[12,3,2,3,5],"c3":[45,4,2],"c4":[4,3,2,3,3,41,1,],"c6":[30]}
#a2 = {"c0":[12,3,3,5],"c3":[44,2],"c4":[4,2,3,31,],"c6":[30]}
#a3 = {"c0":[123,3,5],"c3":[442],"c4":[4,3,31,],"c6":[]}
#singleClassAttr(a1,title="loll")
#doubleClassAttr(a1,a2)
#multiClassAttr([a1,a2,a3],l)
