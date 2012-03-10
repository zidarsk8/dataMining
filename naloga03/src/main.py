#!/usr/bin/python2
import data
import sys
from collections import Counter
from itertools import chain as Chain

d,ds = data.getDataArray()
l = data.getLabelsArray()
stPrimerov = len(l)
stRazredov = max(map(max,l))+1
stAtributov = max(map(max,d))+1

print stRazredov
print stPrimerov
print stAtributov

#print d
#print ds

lc = Counter(Chain(*l))
print lc


#stats = [[[0,ds[i][0],0,ds[i][1],0] for j in ds] for i in xrange(stRazredov)]

stats = {}

for p in xrange(stPrimerov):
	for attr,val in d[p].items():
		for clas in l[p]:
			if not stats.has_key(clas):
				stats[clas] = {}
			if not stats[clas].has_key(attr):
				stats[clas][attr] = [0,0,0]
			stats[clas][attr][0] += val
			stats[clas][attr][1] += 1


#for i,j in enumerate(stats):
#	print j
#	for a in stats[j]:
#		print "     ",a,stats[j][a],ds[a]
#

at = 20
cou = 0
for i,a in enumerate(stats[at]):
	if ds[a][1] > 1 and stats[at][a][1] > ds[a][1]*0.8:
		cou += 1
		print i,a,"  ",stats[at][a],"   ",ds[a]

print lc[at],cou

#print ds[9536]
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
