import cPickle
import numpy as np
import sys
import Orange
import random
import infoGain




print "loading data"
data = Orange.data.Table("data/train.tab")

X, y, _ = data.to_numpy()
# m = rows, n = columns
m,n = X.shape 
folds = 10
trees = 4
method = "rf_bin"

cv_ind = [int(float(i)/m*folds) for i in range(m)]
random.seed(12345)
random.shuffle(cv_ind)

yPred = list(cv_ind)

trainD = data.select(cv_ind,1)
testD = data.select(cv_ind,2)


trees=50
permutations=500
nonzero=50

X, y, _ = trainD.to_numpy()

reload(infoGain)

binVal,gains = infoGain.getGains(X, y, permutations, nonzero)





