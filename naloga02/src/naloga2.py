import Orange
import random
import numpy
import jrs
import collections
import itertools
from sets import Set


tl = open("minidata/trainingLabels.csv")

labels = [i.strip().split(",") for i in tl.readlines()]
labelCounter=collections.Counter(itertools.chain(*labels))

a = collections.defaultdict(list)
[a[j].append(i) for i,j in labelCounter.most_common()]


mld=jrs.Data(discretized=True)

for count,labels in a.items():
	print count,labels
	for label in labels:
		print label

	#todo .. ALL THE THINGS
