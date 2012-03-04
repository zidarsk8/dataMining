from os.path import isfile
from random import random
from random import shuffle
from math import log
from matplotlib import pyplot
import numpy 
import sys
import pickle
import Orange
import numpy
import jrs

def classAttr(arr):

	plt = [len(arr["c"+str(y)]) for y in  sorted([int(x[1:]) for x in arr])]
	for y in  sorted([int(x[1:]) for x in arr]):
		print ("c"+str(y)),len(arr["c"+str(y)])


	menMeans = [random()*1000 for x in range(83)]
	womenMeans = [x*0.7 for x in menMeans]

	ind = numpy.arange(len(menMeans)) 
	width = 1

	fig = pyplot.figure()
	ax = fig.add_subplot(111)

	rects1 = ax.bar(ind, menMeans, width, color='b')
	rects2 = ax.bar(ind, womenMeans, width, color='r')

	ax.set_ylabel('Stevilo znacilk')
	ax.set_title('Scores by group and gender')
	ax.set_xlabel("Razred")

	ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

	pyplot.show()
