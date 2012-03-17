import matplotlib.pyplot as plot
import data
from sets import Set
from itertools import chain
from collections import Counter
from random import shuffle


rawL = data.getLabelsArray()
rawD = data.getDataArray()
remLD, remLL = data.removeLeastCommonData(rawD, rawL,5)
remMD, remML = data.removeMostCommonData(rawD, rawL,20)
addD , addL  = data.addFakeData(rawD, rawL,50)

#expD, expL = data.removeLeastCommonData(rawD, rawL,5)
#expD, expL = data.removeMostCommonData(expD, expL,20)
#expD, expL = data.addFakeData(expD, expL,80)

expD, expL = data.addFakeData(rawD, rawL,80)
expD, expL = data.removeLeastCommonData(expD, expL,5)
expD, expL = data.removeMostCommonData(expD, expL,20)

#plot.hist(list(chain(*addL)),bins=83)
#plot.hist(list(chain(*rawL)),bins=83)
#plot.hist(list(chain(*remLL)),bins=83)
#plot.hist(list(chain(*remML)),bins=83)
#plot.hist(list(chain(*expL)),bins=83)
#plot.xlabel("stevilo oznak")
#plot.ylabel("stevilo primerov")
#plot.show()
#plot.close()


f = file("filteredLabels.csv","w")
f.write("\n".join([",".join([str(x) for x in i]) for i in expL ]))
f.flush()
f.close()

f = file("filteredData.csv","w")
f.write("\n".join(["\t".join([str(x) for x in i]) for i in expD ]))
f.flush()
f.close()
