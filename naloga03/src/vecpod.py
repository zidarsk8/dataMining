import itertools
import matplotlib.pyplot as plot
from collections import Counter
import data


labels = data.getLabelsArray()
#rawData = data.getDataArray()
stPrimerov = len(labels)

#bad = data.getBadAttributes(rawData,10)
#rawData = data.filterArr(rawData,bad)

c = Counter(itertools.chain(*labels))
lc = Counter.most_common(c)

dlc = {}
for l in lc:
	dlc[l[0]] = l[1]

#teze = [sum([ dlc[y]**2 for y in x])  for x in labels]
teze = [max([ dlc[y] for y in x])  for x in labels]

teze = sorted([(y,x) for x,y in enumerate(teze)])

ll = labels[:]
povecaj = 100;
counter = 0
for i,j in teze:
	if counter%3 == 0:
		povecaj -= 1
	for x in xrange(povecaj):
		labels.append(ll[j])
	




plot.hist(list(itertools.chain(*labels)),bins=83)
plot.hist(list(itertools.chain(*ll)),bins=83)
plot.xlabel("stevilo oznak")
plot.ylabel("stevilo primerov")
plot.show()
plot.close()

