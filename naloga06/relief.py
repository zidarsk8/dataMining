import Orange
import cPickle
import numpy as np
import functions

uniques = 10 #meja za zveznost
data = Orange.data.Table("data/train.tab")
trainD,y,_ = data.to_numpy()
m,n = trainD.shape
y = y.astype(int)

razlicnih = np.asarray([np.unique(trainD[:,i]).size for i in xrange(n)])
zvezni = [i for i,x in enumerate(razlicnih) if x > uniques]

trainD = trainD[:,zvezni]
preslikavaIndex = [(i,j) for i,j in enumerate(zvezni)]

data = functions.listToOrangeSingleClass(trainD, y)
trainD,y,_ = data.to_numpy()
m,n = trainD.shape
razlicnih = np.asarray([np.unique(trainD[:,i]).size for i in xrange(n)])

reliefScore = {}
for attr in data.domain.attributes:
    reliefScore[attr] = Orange.feature.scoring.Relief(attr, data)
    print attr,reliefScore[attr]

cPickle.dump({"ind":preslikavaIndex,"rel":reliefScore},\
             open("relief_score_continous_filter_uniques_%d.pkl" % uniques,"w"))


