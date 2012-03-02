import Orange
import os.path
import sys
import random
import numpy
import jrs
import collections
import itertools
import pickle
from sets import Set


print "\n\nloading all valid labels"
mldRaw = pickle.load(open("minidata/trainingDataOD.pickled"))
validLabels = [i.name for i in mldRaw.domain.class_vars]

attributeNames = [i.name for i in  mldRaw.domain.features]

print "\ngenerating label counters"
tl = open("minidata/trainingLabels.csv")
labels = [i.strip().split(",") for i in tl.readlines()]
labelCounter=collections.Counter(itertools.chain(*labels))

print "\nfiltering unused labels"
a = collections.defaultdict(list)
[a[j].append("c"+str(i)) for i,j in labelCounter.most_common() if "c"+str(j) in validLabels]


print "\nloading mld\n"
mld=jrs.Data(discretized=True)

#data = mld.get_single_class_data("c40")



originalGains = {}
if os.path.isfile('minidata/gainTables.pickled'):

#	numCounters = len(a.items())
#	curCounter = 1
#	for count,labels in a.items():
#	
#		prependStr = str(curCounter)+" / "+str(numCounters)
#		curCounter += 1
#		#print count,labels
#		mld.get_single_class_data("c40")
#		
#		for label in labels:
#			data = mld.get_single_class_data(label)
#			
#			originalGains[label] = {}
#			indexCount = len(attributeNames)
#			for index,attr in enumerate(attributeNames):
#				sys.stdout.write(prependStr+"    InfoGain: %d%%   \r" % (index*100/indexCount) )
#				originalGains[label][attr] = Orange.feature.scoring.InfoGain(attr,data)
#	
#	pickle.dump("minidata/gainTables.pickled")
#	
	print "\n"


#	prvotnGain = Orange.feature.scoring.InfoGain(data.domain.features[0],data)
#	
#	a = [x.get_class().value for x in data]
#	a_original = list(a)
#	random.shuffle(a)
#	
#	for i,ex in enumerate(data):
#		ex.set_class(a[i])
#	
#	randomGain =  Orange.feature.scoring.InfoGain(data.domain.features[0],data)
#
#	for i,ex in enumerate(data):
#		ex.set_class(a_original[i])


#print mld.domain.features
#print mld.domain.class_var


#print mld[0][mld.domain.features[0]]

	#todo .. ALL THE THINGS
