#!/usr/bin/python2
import data
import sys
from collections import Counter
from itertools import chain as Chain

d,ds = data.getDataArray()
l = data.getLabelsArray()
lc = Counter(Chain(*l))
stPrimerov = len(l)

#print stPrimerov
#print d
#print ds
#print lc


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

# ta kos kode pokaze da stevilo razredov ni odvisno od
# stevila nenicelnih atributov, saj je to povprecje enako
# pri vseh stevilih razredov za posamezni primer.
#lens = {}
#for p in xrange(stPrimerov):
#	if not lens.has_key(len(l[p])):
#		lens[len(l[p])] = [0,0]
#	lens[len(l[p])][0] += len(d[p])
#	lens[len(l[p])][1] += 1
#
#for i,j in lens.items():
#	print i,j, j[0]/j[1]



#at = 20
#cou = 0
#for i,a in enumerate(stats[at]):
#	if ds[a][1] > 1 and stats[at][a][1] > ds[a][1]*0.8:
#		cou += 1
#		print i,a,"  ",stats[at][a],"   ",ds[a]
#
#print lc[at],cou
#
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
