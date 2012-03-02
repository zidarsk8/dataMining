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
numberOfLines = len(labels)
labelCounter=collections.Counter(itertools.chain(*labels))

print "\nfiltering unused labels"
labelGroups = collections.defaultdict(list)
[labelGroups[j].append("c"+str(i)) for i,j in labelCounter.most_common() if "c"+str(j) in validLabels]


print "\nloading mld\n"
mld=jrs.Data(discretized=True)

#data = mld.get_single_class_data("c40")

infoGain = Orange.feature.scoring.InfoGain()

originalGains = {}
if os.path.isfile('minidata/gainTables.pickled'):
	print "reading data from pickle file"
	originalGains = pickle.load(open("minidata/gainTables.pickled"))
else:
	print ""	
	labelCount = len(validLabels)
	for lc, label in enumerate(validLabels):
		data = mld.get_single_class_data(label)
		originalGains[label] = {}
		indexCount = len(attributeNames)
		for index,attr in enumerate(attributeNames):
			sys.stdout.flush()
			sys.stdout.write("Calculating InfoGain for all classes: %d%%   \r" % (1+index*100/indexCount/labelCount + 100*lc/labelCount) )
			originalGains[label][attr] = infoGain(attr,data)

	print "\nDumping original InfoGains into a pickle file"
	pickle.dump(originalGains,file("minidata/gainTables.pickled","w"),-1)


shuffles = 10;
print "Calculating shuffled InfoGain ",shuffles," times"


randomGains = {}
if os.path.isfile('minidata/randomGains.pickled'):
	print "reading random gains from pickle file"
	randomGains = pickle.load(open("minidata/randomGains .pickled"))
else:
	for group, labels in labelGroups.items():
		randomGains[group] = {}
		for attr in attributeNames:
			randomGains[group][attr] = []

for group, labels in labelGroups.items():

	print "\n\nCalculating random gains for classes ",labels
	data = mld.get_single_class_data(labels[0])
	a = ['F']*(numberOfLines - group) + ["T"]*group
	for i in range(shuffles):
		sys.stdout.flush()
		sys.stdout.write("Calculating random InfoGain for all classes: %d%%   \r" % (100*(i+1)/shuffles) )

		random.shuffle(a)
		[ex.set_class(a[i]) for i,ex in enumerate(data)]

		[randomGains[group][attr].append(infoGain(attr,data)) for attr in attributeNames]
		#[randomGains[group][attr].append(infoGain(attr,data)) for attr in data.domain.features]



#	prvotnGain = Orange.feature.scoring.InfoGain(data.domain.features[0],data)
#	
#	
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
