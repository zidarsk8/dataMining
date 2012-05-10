from multiprocessing import Pool
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

def randomForest(trainD,testD,args={}):
    min_instances = args["min_instances"] if type(args) == dict and args.has_key("min_instances") else 5
    max_depth = args["max_depth"] if type(args) == dict and args.has_key("max_depth") else 100
    trees = args["trees"] if type(args) == dict and args.has_key("trees") else 100
    
    st = Orange.classification.tree.SimpleTreeLearner(min_instances=min_instances, max_depth=max_depth)

    rfs = Orange.ensemble.forest.RandomForestLearner(trees=trees, name="rfs", base_learner=st)
    #rfs = Orange.ensemble.forest.RandomForestLearner(trees=trees, name="rfs")
    return getProb(rfs, trainD, testD)
    

def randomForestBin(trainD,testD,args):
    trees = args["trees"] if type(args) == dict and args.has_key("trees") else 100
    permutations = args["permutations"] if type(args) == dict and args.has_key("permutations") else 1000
    nonzero = args["nonzero"] if type(args) == dict and args.has_key("nonzero") else 20
    duplicateCount = args["duplicateCount"] if type(args) == dict and args.has_key("duplicateCount") else 500

    trainX, trainy, _ = trainD.to_numpy()
    testX, testy, _ = testD.to_numpy()
    
    if type(duplicateCount) == float:
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
    _,yTrue,_ = trainD.to_numpy()
    return np.array([float(yTrue.sum())/yTrue.size]*len(testD))

def logLoss(yTrue,yPred):
    if len(yTrue) != len(yPred) : return -1
    N = len(yTrue)
    yTrue = np.array(yTrue)
    yTPred = np.array([x for i,x in enumerate(yPred) if yTrue[i]==1])
    yFPred = np.array([x for i,x in enumerate(yPred) if yTrue[i]==0])
    return -1.0/N *( sum(np.log(yTPred)) + sum(np.log(1-yFPred)) )


def crosval(data,method="rf",indexes=0,folds=10,status=False,threads=1,args={}):
    #pripravimo indekse za krosvalidacijo in list za rezultate
    m = len(data)
    folds = min(100,max(folds,2))
    if not isinstance(indexes,list) or len(indexes) != m:
        indexes = [int(float(i)/m*folds) for i in range(m)]
        random.shuffle(indexes)
    yPred = list(indexes)
    
    def cv(fold):
        if status and threads == 1: 
            sys.stdout.write("\r%s crossvalidation: %d/%d" %(method,fold+1,folds))
            sys.stdout.flush()
        trainD = data.select(indexes,fold,negate=1)
        testD = data.select(indexes,fold)
        if method ==  "rf_bin"  : return randomForestBin(trainD, testD, args)
        elif method == "rf"     : return randomForest(trainD, testD, args)
        elif method == "svm"    : return svm(trainD, testD)
        elif method == "knn"    : return knn(trainD, testD)
    
    if threads>1:
        p = Pool(processes=2)
        rr = p.map(cv, range(folds))
    else:
        rr = map(cv,range(folds))

    for fold in range(folds):
        yPred = [i if i!= fold else rr[fold].pop(0)*0.9998+0.0001 for i in yPred]
        
    #if status and threads == 1: print ""
    return yPred


if __name__ == "__main__":
    #random.seed(123)
    print "loading data"
    data = Orange.data.Table("data/train.tab")
    X,y,_ = data.to_numpy()
    a = range(X.shape[1])
    random.shuffle(a)
    X = X[:1000,a[:400]]
    y = y[:1000]
    data = functions.listToOrangeSingleClass(X, y.astype(int))
    
    #data = cPickle.load(file("data/minidata400x200.pkl"))
    #X,y,_ = data.to_numpy()
    
    folds = 10
    method = "rf"
    ll = {i:[] for i in range(10,401,10)}
    for j in range(50):
        print "----------------------------------------------"
        for i in range(10,401,10):
            args = {"max_depth":i,
                    "trees":50}
            yPred = crosval(data, method=method, folds=folds, status=True,args=args);
            ll[i].append(logLoss(y,yPred))
            print "\r",i,"corssval:", ll[i],"                    "
        
    #cPickle.dump(yPred,open("%s_cv_%d_ll1000_%d.pkl" % (method,folds,ll*1000) ,"w"))
    
