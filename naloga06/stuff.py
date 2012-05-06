import cPickle
import numpy as np
import sys
import Orange
import random
import infoGain




def listToOrangeSingleClass(X,y):
    features = [Orange.feature.Continuous("%d" % i) for i in range(len(X[0]))]
    class_var = Orange.feature.Discrete("class", values=["0","1"])
    domain = Orange.data.Domain(features + [class_var])
    data = Orange.data.Table(domain)
    [data.append(Orange.data.Instance(domain, list(X[i])+[["0", "1"][y[i]]])) for i in range(len(X))]
    return data


if __name__ == "__main__":
    print "loading data"
    data = Orange.data.Table("data/train.tab")
    
    X, y, _ = data.to_numpy()
    m,n = X.shape 
    folds = 5
    trees = 20
    method = "knn"
    
    cv_ind = [int(float(i)/m*folds) for i in range(m)]
    
    yPred = list(cv_ind)
    #yPred = []
    for fold in range(folds):
        sys.stdout.write("\r%s crossvalidation: %d/%d" %(method,fold+1,folds))
        sys.stdout.flush()
        trainD = data.select(cv_ind,fold,negate=1)
        testD = data.select(cv_ind,fold)
        
        
        
        yPred += randomForestBin(trainD, testD, trees, duplicateCount = 2)        

    yPred = np.array(yPred)*0.9998+0.0001
    
    #yPred = np.array([y.sum()/y.size]*m)
    print ""
    print method,"logLoss: ", logLoss(y, yPred)
    
    #cPickle.dump(yPred,open("%s_cv_%d_ll1000_%d.pkl" % (method,folds,ll*1000) ,"w"))
