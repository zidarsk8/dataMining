import matplotlib.pyplot as plot
import numpy as np
import sys
import Orange
import random

def randomForest(trainD,testD,trees=50):
    rf = Orange.ensemble.forest.RandomForestLearner(trees=trees, name="forest")
    cl = rf(trainD)
    return [cl(t, Orange.classification.Classifier.GetProbabilities)[True] for t in testD]

def svm(trainD,testD):
    s = Orange.classification.svm.LinearSVMLearner(name="linearSVM")
    cl = s(trainD)
    return [cl(t, Orange.classification.Classifier.GetProbabilities)[True] for t in testD]

def logLoss(yTrue,yPred):
    if len(yTrue) != len(yPred) : return -1
    N = len(yTrue)
    yTrue = np.array(yTrue)
    yPred = np.array(yPred)
    return -1.0/N * sum( yTrue*np.log(yPred) + (1-yTrue)*np.log(1-yPred) )


print "loading data"
data = Orange.data.Table("data/train.tab")

X, y, _ = data.to_numpy()
# m = rows, n = columns
m,n = X.shape 
folds = 10


cv_ind = [int(float(i)/m*folds) for i in range(m)]
random.seed(12345)
random.shuffle(cv_ind)

yPred = list(cv_ind)
#yPred = []
for fold in range(folds):
    sys.stdout.write("\rcrossvalidation: %d/%d" %(fold+1,folds))
    sys.stdout.flush()
    trainD = data.select(cv_ind,fold,negate=1)
    testD = data.select(cv_ind,fold)
    #yPred += randomForest(trainD, testD, 100)
    rr = randomForest(trainD, testD, 10)
    ind = [i for i,j in enumerate(cv_ind) if j == fold]
    for i,r in enumerate(rr):
        yPred[ind[i]] = r

yPred = [x if x>0 else 0.00001 for x in yPred]
yPred = [x if x<1 else 0.99999 for x in yPred]
#yPred = yPred*0.98+0.01

_,yTrue,_ = data.to_numpy()
print "rf done"

print "logLoss rf: ", logLoss(yTrue, yPred);
