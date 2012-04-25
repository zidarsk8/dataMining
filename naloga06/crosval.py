import cPickle
import numpy as np
import sys
import Orange
import random

def getProb(lrn,trainD,testD):
    cl = lrn(trainD)
    return [cl(t, Orange.classification.Classifier.GetProbabilities)[True] for t in testD]

def randomForest(trainD,testD,trees=50):
    s = Orange.ensemble.forest.RandomForestLearner(trees=trees, name="forest")
    return getProb(rf, trainD, testD)

def svm(trainD,testD):
    lsvm = Orange.classification.svm.LinearSVMLearner(name="linearSVM")
    return getProb(lsvm, trainD, testD)

def knn(trainD,testD):
    kn = Orange.classification.knn.kNNLearner(name="knn")
    return getProb(kn, trainD, testD)

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

yPred = list(cv_ind)
#yPred = []
for fold in range(folds):
    sys.stdout.write("\r%s crossvalidation: %d/%d" %(method,fold+1,folds))
    sys.stdout.flush()
    trainD = data.select(cv_ind,fold,negate=1)
    testD = data.select(cv_ind,fold)
    #rr = randomForest(trainD, testD, trees)
    #rr = svm(trainD, testD)
    rr = knn(trainD, testD)
    ind = [i for i,j in enumerate(cv_ind) if j == fold]
    for i,r in enumerate(rr):
        yPred[ind[i]] = r

yPred = [x if x>0 else 0.0001 for x in yPred]
yPred = [x if x<1 else 0.9999 for x in yPred]
#yPred = yPred*0.98+0.01

_,yTrue,_ = data.to_numpy()
ll = logLoss(yTrue, yPred)
print method,"logLoss: ", ll
cPickle.dump(yPred,open("%s_cv_%d_ll1000_%d.pkl" % (method,folds,ll*1000) ,"w"))
