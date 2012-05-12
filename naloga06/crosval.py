from multiprocessing import Pool
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
    min_instances = args["min_instances"] if type(args) == dict and\
            args.has_key("min_instances") else 5
    max_depth = args["max_depth"] if type(args) == dict and\
            args.has_key("max_depth") else 100
    trees = args["trees"] if type(args) == dict and\
            args.has_key("trees") else 100
    st = Orange.classification.tree.SimpleTreeLearner(\
            min_instances=min_instances, max_depth=max_depth)
    rfs = Orange.ensemble.forest.RandomForestLearner(\
            trees=trees, name="rfs", base_learner=st)
    return getProb(rfs, trainD, testD)
    

def randomForestBin(trainD,testD,args):
    permutations = args["permutations"] if type(args) == dict and\
            args.has_key("permutations") else 1000
    nonzero = args["nonzero"] if type(args) == dict and\
            args.has_key("nonzero") else 20
    duplicateCount = args["duplicateCount"] if type(args) == dict and\
            args.has_key("duplicateCount") else 0.5

    if duplicateCount > 0:
        print "making duplicates"
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
        print "making duplicates",X.shape

    return randomForest(trainD, testD, args)

def rfInnerCrossVal(trainD,testD,args):
    if type(args) != dict : args = {}
    method = "rf_bin"
    folds = 5
    repeat = 10
    _,y,_ = trainD.to_numpy()
    
    
    if not args.has_key("min_instances"):
        ran = range(3,16,1)
        ll = {i:[] for i in ran}
        for j in range(20):
            print "searching for best min_instances (%d/%d) ",(j,repeat)
            for i in ran:
                a = {"min_instances":i, "trees":50}
                yPred = crosval.crosval(trainD, method=method, folds=folds, status=False, args=a, threads=1);
                ll[i].append(crosval.logLoss(y,yPred))
        best = sorted([(sum(j)/len(j),i)for i,j in ll.items()])[0][1]
        args["min_instances"] = best
    
    return randomForestBin(trainD, testD, args);

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
    m = len(data)
    folds = min(100,max(folds,2))
    if not isinstance(indexes,list) or len(indexes) != m:
        indexes = [int(float(i)/m*folds) for i in range(m)]
        random.shuffle(indexes)
        
    cvfolds = [{"data":data,"fold":x,"indexes":indexes,"method":method,\
                "args":args,"status":status} for x in range(folds)]
    
    def cv(d):
        if d["status"]: 
            sys.stdout.write("\r%s crossvalidation fold: %d" %(d["method"],d["fold"]+1))
            sys.stdout.flush()
        trainD = d["data"].select(d["indexes"],d["fold"],negate=1)
        testD = d["data"].select(d["indexes"],d["fold"])
        if d["method"] ==  "rf_bin"  : return randomForestBin(trainD, testD, d["args"])
        elif d["method"] == "rf"     : return randomForest(trainD, testD, d["args"])
        elif d["method"] == "svm"    : return svm(trainD, testD)
        elif d["method"] == "knn"    : return knn(trainD, testD)

    if threads>1:
        p = Pool(processes=threads)
        rr = p.map(cv, cvfolds)
    else:
        rr = map(cv, cvfolds)

    yPred = np.array(indexes)
    for fold in range(folds): 
        yPred[yPred==fold] = rr[fold]
    yPred = yPred*0.9998+0.0001

    if status: print ""
    return yPred


if __name__ == "__main__":
    print "loading data"
    data = Orange.data.Table("data/train.tab")
    _,y,_ = data.to_numpy()

    f = 10
    m = "rf"

    a = {"trees":50,"max_depth":200}
    yp = crosval(data, method=m, folds=f, status=True, args=a,threads=2);
    
    print "corssval:", logLoss(y,yp),"                    "
    

