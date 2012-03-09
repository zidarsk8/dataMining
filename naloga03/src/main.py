#!/usr/bin/python2
import data
import sys

d = data.getDataArray(True)
l = data.getLabelsArray(True)
stPrimerov = len(l)
stRazredov = max(map(max,l))+1
stAtributov = max(map(max,d))+1

print stRazredov
print stPrimerov
print stAtributov


print d

stats = [[[0,0,0,0,0] for j in xrange(stAtributov)] for i in xrange(stRazredov)]

for p in xrange(stPrimerov):
	for attr,val in d[p].items():
		for clas in l[p]:
			stats[clas][attr][0] += val
			stats[clas][attr][2] += 1


#
#for i in xrange(stPrimerov):
#	for a in xrange(stAtributov):
#		for r in l[i]:
#			stats[r][a][0] += d[i][a]
#			stats[r][a][2] += d[i][a]
#



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
