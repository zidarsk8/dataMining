import Orange
import random
import numpy
import jrs
import collections
import itertools
import pickle
from sets import Set


print "\n\ngenerating label counters"
tl = open("minidata/trainingLabels.csv")
labels = [i.strip().split(",") for i in tl.readlines()]
labelCounter=collections.Counter(itertools.chain(*labels))

a = collections.defaultdict(list)
[a[j].append(i) for i,j in labelCounter.most_common()]


print "\nloading mld"
mld=jrs.Data(discretized=True)

data = mld.get_single_class_data("c40")
	


	prvotnGain = Orange.feature.scoring.InfoGain(data.domain.features[0],data)
	
	a = [x.get_class().value for x in data]
	a_original = list(a)
	random.shuffle(a)
	
	for i,ex in enumerate(data):
		ex.set_class(a[i])
	
	randomGain =  Orange.feature.scoring.InfoGain(data.domain.features[0],data)

	for i,ex in enumerate(data):
		ex.set_class(a_original[i])

#mldRaw = pickle.load(open("minidata/trainingDataOD.pickled"))

#print mld.domain.features
#print mld.domain.class_var

#for i in mld.domain.class_vars:
#	print i

#print mld[0][mld.domain.features[0]]

#for count,labels in a.items():
#	print count,labels
#	mld.get_single_class_data("c40")
#	for label in labels:
#		print label
#
#	#todo .. ALL THE THINGS
