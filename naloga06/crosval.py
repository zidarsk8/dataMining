import cPickle
import numpy as np
import sys
import Orange
import random
import infoGain
import functions

def getProb(lrn,trainD,testD):
    cl = lrn(trainD)
    return [cl(t, Orange.classification.Classifier.GetProbabilities)[True] for t in testD]

def randomForest(trainD,testD,trees=50):
    rf = Orange.ensemble.forest.RandomForestLearner(trees=trees, name="forest")
    return getProb(rf, trainD, testD)
    

def randomForestBin(trainD,testD,trees=50,permutations=1000,nonzero=50,duplicateCount=0.2):
    

    trainX, trainy, _ = trainD.to_numpy()
    testX, testy, _ = testD.to_numpy()
    
    if int(duplicateCount) != duplicateCount:
        duplicateCount = int(trainX.shape[1]*duplicateCount);
    binVal,gains = infoGain.getGains(trainX, trainy, permutations, nonzero)
    ind = [x[2] for x in gains[0] if x[1] > nonzero][:duplicateCount]
    meje = [binVal[i] for i in ind]
    trainX = np.concatenate((trainX,(trainX.T[ind].T>meje).astype(float)),axis=1)
    testX = np.concatenate((testX,(testX.T[ind].T>meje).astype(float)),axis=1)

    X = np.concatenate((trainX,testX),axis=0)
    y = np.concatenate((trainy,testy),axis=0)
    data = functions.listToOrangeSingleClass(X, y.astype(int))
    ind = [0]*trainy.size + [1]*testy.size
    trainD = data.select(ind,0)
    testD = data.select(ind,1)

    rf = Orange.ensemble.forest.RandomForestLearner(trees=trees, name="forest")
    return getProb(rf, trainD, testD)

def svm(trainD,testD):
    lsvm = Orange.classification.svm.LinearSVMLearner(name="linearSVM")
    return getProb(lsvm, trainD, testD)

def knn(trainD,testD):
    kn = Orange.classification.knn.kNNLearner(name="knn")
    return getProb(kn, trainD, testD)

def constVal(trainD,testD):
    yPred = np.array([0.1]*m)
    _,yTrue,_ = trainD.to_numpy()
    
    ll = logLoss(yTrue, yPred)
    prev = ll
    for i in range(2000,7000):
        a = i/10000.0
        yPred = np.array([a]*m)
        ll = logLoss(yTrue, yPred)
        print "%.6f   %.10f      %d" % (a,ll, int(prev<ll))
        if (prev<ll):
            break;
        prev = ll

def logLoss(yTrue,yPred):
    if len(yTrue) != len(yPred) : return -1
    N = len(yTrue)
    yTrue = np.array(yTrue)
    yTPred = np.array([x for i,x in enumerate(yPred) if yTrue[i]==1])
    yFPred = np.array([x for i,x in enumerate(yPred) if yTrue[i]==0])
    return -1.0/N *( sum(np.log(yTPred)) + sum(np.log(1-yFPred)) )





if __name__ == "__main__":
    print "loading data"
    data = Orange.data.Table("data/train.tab")
    
    X, y, _ = data.to_numpy()
    # m = rows, n = columns
    m,n = X.shape 
    folds = 10
    trees = 50
    method = "rf_bin"
    
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
        
        if method ==  "rf_bin"  : rr = randomForestBin(trainD, testD, trees, duplicateCount = 0.99)
        elif method == "rf"     : rr = randomForest(trainD, testD, trees)
        elif method == "svm"    : rr = svm(trainD, testD)
        elif method == "knn"    : rr = knn(trainD, testD)
        
        ind = [i for i,j in enumerate(cv_ind) if j == fold]
        for i,r in enumerate(rr):
            yPred[ind[i]] = r

    yPred = np.array(yPred)*0.9998+0.0001
    
    #yPred = np.array([y.sum()/y.size]*m)
    print ""
    print method,"logLoss: ", logLoss(y, yPred)
    
    #cPickle.dump(yPred,open("%s_cv_%d_ll1000_%d.pkl" % (method,folds,ll*1000) ,"w"))
