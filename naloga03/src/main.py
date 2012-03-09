#!/usr/bin/python2
import data
import sys

d = data.getDataArray()
l = data.getLabelsArray()
stPrimerov = len(l)
stRazredov = max(map(max,l))+1
stAtributov = len(d[0])

print stRazredov
print stPrimerov
print stAtributov


stats = [[[0,0,0,0,0]]*stAtributov]*stRazredov

for i in xrange(stPrimerov):
	for a in xrange(stAtributov):
		for r in l[i]:
			stats[r][a][0] += d[i][a]
			stats[r][a][2] += d[i][a]




#for clas in xrange(stRazredov):
#	for attr in xrange(stAtributov):
#		if attr%10 == 0:
#			sys.stdout.flush()
#			sys.stdout.write("\rGenerating statistics: %3.1f%%" % \
#				(100.0*(clas*stAtributov+attr)/(stRazredov*stAtributov)))
#		for i in xrange(stPrimerov):
#			if l[clas][i]:
#				stats[clas][attr][0] += d[i][attr]
#				stats[clas][attr][2] += d[i][attr]==0
#			else:
#				stats[clas][attr][1] += d[i][attr]
#				stats[clas][attr][3] += d[i][attr]==0
#
for i,j in enumerate(stats[40]):
	print i,j
