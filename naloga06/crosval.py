import cPickle
import numpy as np
import sys
import Orange
import random

def getProb(lrn,X,testD):
    cl = lrn(X)
    return [cl(t, Orange.classification.Classifier.GetProbabilities)[True] for t in testD]

def randomForest(X,testD,trees=50):
    s = Orange.ensemble.forest.RandomForestLearner(trees=trees, name="forest")
    return getProb(rf, X, testD)

def svm(X,testD):
    lsvm = Orange.classification.svm.LinearSVMLearner(name="linearSVM")
    return getProb(lsvm, X, testD)

def knn(X,testD):
    kn = Orange.classification.knn.kNNLearner(name="knn")
    return getProb(kn, X, testD)

def logLoss(yTrue,yPred):
    if len(yTrue) != len(yPred) : return -1
    N = len(yTrue)
    yTrue = np.array(yTrue)
    yPred = np.array(yPred)
    return -1.0/N * sum( yTrue*np.log(yPred) + (1-yTrue)*np.log(1-yPred) )

#def logLoss(yTrue,yPred):
#    if len(yTrue) != len(yPred) : return -1
#    N = len(yTrue)
#    yTrue = np.array(yTrue)
#    yTPred = np.array([x for i,x in enumerate(yPred) if yTrue[i]==1])
#    yFPred = np.array([x for i,x in enumerate(yPred) if yTrue[i]==0])
#    return -1.0/N *( sum(np.log(yTPred)) + sum(np.log(1-yFPred)) )


print "loading data"
data = Orange.data.Table("data/train.tab")

X, y, _ = data.to_numpy()
# m = rows, n = columns
m,n = X.shape 
folds = 10
trees = 100
method = "knn"

cv_ind = [int(float(i)/m*folds) for i in range(m)]
random.seed(12345)
random.shuffle(cv_ind)

#yPred = list(cv_ind)
##yPred = []
#for fold in range(folds):
#    sys.stdout.write("\r%s crossvalidation: %d/%d" %(method,fold+1,folds))
#    sys.stdout.flush()
#    X = data.select(cv_ind,fold,negate=1)
#    testD = data.select(cv_ind,fold)
#    #rr = randomForest(X, testD, trees)
#    #rr = svm(X, testD)
#    rr = knn(X, testD)
#    ind = [i for i,j in enumerate(cv_ind) if j == fold]
#    for i,r in enumerate(rr):
#        yPred[ind[i]] = r
#
#yPred = np.array(yPred)*0.9998+0.0001

yPred = np.array([0.1]*m)
_,yTrue,_ = data.to_numpy()

ll = logLoss(yTrue, yPred)
prev = ll
best = 0
for i in range(2000,7000):
    a = i/10000.0
    yPred = np.array([a]*m)
    ll = logLoss(yTrue, yPred)
    print "%.6f   %.10f      %d" % (a,ll, int(prev<ll))
    if (prev<ll):
        best = a
        break;
    prev = ll
    

print method,"logLoss: ", ll

#cPickle.dump(yPred,open("%s_cv_%d_ll1000_%d.pkl" % (method,folds,ll*1000) ,"w"))
