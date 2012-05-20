import matplotlib.pyplot as plt
import numpy as np
import functions
import crosval
import pickle
import Orange
import random

data = Orange.data.Table("data/train.tab")
X,y,_ = data.to_numpy()


#random.seed(123)
#a = range(X.shape[1])
#random.shuffle(a)
#X = X[:300,a[:300]]
#y = y[:300]
#data = functions.listToOrangeSingleClass(X, y.astype(int))
##data = cPickle.load(file("data/minidata400x200.pkl"))
#X,y,_ = data.to_numpy()

folds = 10
method = "rf_bin"
f = 0
t = X.shape[1]
step = (t-f)/30 
ran = range(f,t+1,step)
ll = {i:[] for i in ran}
for j in range(20):
    print "----------------------",j,"------------------------"
    for i in ran:
        args = {"permutations":500,
                "duplicateCount":i,
                "nonzero":10,
                "trees":50}
        yPred = crosval.crosval(data, method=method, folds=folds, status=False, args=args, threads=2);
        ll[i].append(crosval.logLoss(y,yPred))
        print i,"corssval:", ll[i][-1],"            "
        

pickle.dump(ll,open("cv_%s_%d_%d_folds_%d_duplicateCount_%d-%d-%d.pkl" % (method,X.shape[0],X.shape[1],folds,f,t,step),"w"))

aa = np.array(sorted([(i, min(j),sum(j)/len(j),max(j)) for i,j in ll.items()]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(aa[:,0],aa[:,1],"-")
ax.plot(aa[:,0],aa[:,2],"-")
ax.plot(aa[:,0],aa[:,3],"-")

plt.show()
