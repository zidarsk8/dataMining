#!/usr/bin/python2
import data
import sys

d = data.getDataArray()
l = data.getLabelsArray()
stPrimerov = len(l)
stRazredov = len(l[0])
stAtributov = len(d[0])

print stRazredov
print stPrimerov
print stAtributov


stats = [[[0,0,0,0,0]]*stAtributov]*stPrimerov
for clas in xrange(stRazredov):
	for attr in xrange(stAtributov):
		if attr%10 == 0:
			sys.stdout.flush()
			sys.stdout.write("\rGenerating statistics: %3.1f%%" % \
				(100.0*(clas*stAtributov+attr)/(stRazredov*stAtributov)))
		for i in xrange(stPrimerov):
			stats[clas][attr][0] += d[i][attr]
