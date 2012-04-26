import Orange
import cPickle
import numpy as np
import functions

uniques = 10 #meja za zveznost
data = Orange.data.Table("data/train.tab")
X,y,_ = data.to_numpy()
m,n = X.shape
y = y.astype(int)

razlicnih = np.asarray([np.unique(X[:,i]).size for i in xrange(n)])
zvezni = [i for i,x in enumerate(razlicnih) if x > uniques]

X = X[:,zvezni]
preslikavaIndex = [(i,j) for i,j in enumerate(zvezni)]

data = functions.listToOrangeSingleClass(X, y)
X,y,_ = data.to_numpy()
m,n = X.shape
razlicnih = np.asarray([np.unique(X[:,i]).size for i in xrange(n)])

reliefScore = {}
for attr in data.domain.attributes:
    reliefScore[attr] = Orange.feature.scoring.Relief(attr, data)
    print attr,reliefScore[attr]

cPickle.dump({"ind":preslikavaIndex,"rel":reliefScore},\
             open("relief_score_continous_filter_uniques_%d.pkl" % uniques,"w"))


