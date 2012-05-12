import matplotlib.pyplot as plt
import numpy as np
import functions
import crosval
import pickle
import Orange
import random

data = Orange.data.Table("data/train.tab")
X,y,_ = data.to_numpy()


random.seed(123)
a = range(X.shape[1])
random.shuffle(a)
X = X[:400,a[:400]]
y = y[:400]
data = functions.listToOrangeSingleClass(X, y.astype(int))
#data = cPickle.load(file("data/minidata400x200.pkl"))
X,y,_ = data.to_numpy()

folds = 10
method = "rf"

ran = range(0,401,20)
ll = {i:[] for i in ran}
for j in range(10):
    print "----------------------",j,"------------------------"
    for i in ran:
        args = {"permutations":400,
                "duplicateCount":0,
                "nonzero":5,
                "trees":20}
        yPred = crosval.crosval(data, method=method, folds=folds, status=False, args=args, threads=1);
        ll[i].append(crosval.logLoss(y,yPred))
        print i,"corssval:", ll[i][-1],"            "
        

#pickle.dump(ll,open("ll_1700_1000_min_in_1_26_2.pkl","w"))

aa = np.array(sorted([(i, min(j),sum(j)/len(j),max(j)) for i,j in ll.items()]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(aa[:,0],aa[:,1],"-")
ax.plot(aa[:,0],aa[:,2],"-")
ax.plot(aa[:,0],aa[:,3],"-")

plt.show()
