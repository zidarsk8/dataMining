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





plot.hist(list(chain(*addL)),bins=83)
plot.hist(list(chain(*rawL)),bins=83)
plot.hist(list(chain(*remLL)),bins=83)
plot.hist(list(chain(*remML)),bins=83)
plot.xlabel("stevilo oznak")
plot.ylabel("stevilo primerov")
plot.show()
plot.close()

