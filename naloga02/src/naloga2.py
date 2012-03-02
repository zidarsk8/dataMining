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
if os.path.isfile('minidata/originalGains.pickled'):
	print "reading data from pickle file"
	originalGains = pickle.load(open("minidata/originalGains.pickled"))
else:
	print ""	
	labelCount = len(validLabels)
	for lc, label in enumerate(validLabels):
		data = mld.get_single_class_data(label)
		originalGains[label] = {}
		indexCount = len(attributeNames)
		for index,attr in enumerate(attributeNames):
			sys.stdout.flush()
			sys.stdout.write("\rCalculating InfoGain for all classes: %3d%%" % (1+index*100/indexCount/labelCount + 100*lc/labelCount) )
			originalGains[label][attr] = infoGain(attr,data)

	print "\nDumping original InfoGains into a pickle file"
	pickle.dump(originalGains,file("minidata/originalGains.pickled","w"),-1)


randomGains = {}
if os.path.isfile('minidata/randomGains.pickled'):
	print "reading random gains from pickle file"
	randomGains = pickle.load(open("minidata/randomGains.pickled"))
else:
	for group, labels in labelGroups.items():
		randomGains[group] = {}
		for attr in attributeNames:
			randomGains[group][attr] = []

shuffles = 10;
print "Calculating shuffled InfoGain ",shuffles," times"

numOfGroups = len(labelGroups)
countGroups = 0
for group, labels in labelGroups.items():
	#print "\n\nCalculating random gains for classes ",labels
	data = mld.get_single_class_data(labels[0])
	a = ['F']*(numberOfLines - group) + ["T"]*group
	for i in range(shuffles):
		sys.stdout.flush()
		curProc = 100*(i+1)/shuffles
		fullProc = countGroups*100/numOfGroups + curProc/numOfGroups +1
		sys.stdout.write("\r%3d%% - Class group: (%2d/%2d) len: %4d : %3d%%" % (fullProc,countGroups+1,numOfGroups,len(randomGains[group][attributeNames[0]]),curProc))

		random.shuffle(a)
		[ex.set_class(a[i]) for i,ex in enumerate(data)]

		[randomGains[group][attr].append(infoGain(attr,data)) for attr in attributeNames]
	countGroups += 1

print "saving new randomGains pickle file"
pickle.dump(randomGains,file("minidata/randomGains.pickled","w"),-1)

print "sorting randomGains"
[a.sort() for x,i in randomGains.items() for y,a in i.items()]

alpha = 0.05

relevantAttributes = {}
for group, labels in labelGroups.items():
	for label in labels:
		relevantAttributes[label] = []
		[relevantAttributes[label].append(attr) for attr in attributeNames if (randomGains[group][attr][int((1-alpha)*(len(randomGains[group][attr])-1))] < originalGains[label][attr])]
				
for a,b in relevantAttributes.items():
	print a,b

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
