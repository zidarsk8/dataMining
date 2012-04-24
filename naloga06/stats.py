import matplotlib.pyplot as plot
import numpy as np
import Orange


data = Orange.data.Table("data/train.tab");



X, y, _ = data.to_numpy();

print "stevilo primerov(m) %d\nstevilo atributov(n) %d" % X.shape

print "stevilo napovedanih razredov %d/%d  oziroma %.1f%%" %\
		(sum(y),len(y),(float(sum(y))/len(y)*100.0))

m,n = X.shape
print "stevilo nul v celotni matriki %d/%d oziroma %.1f%%" %\
		(sum(sum(X==0)),(n*m) , sum(sum(X==0))*100.0/(m*n))

nenicelnih = sum(X != 0);
plot.hist(nenicelnih,bins=50)
plot.xlabel("st. nenicelnih vrednosti")
plot.ylabel("st. atributov")
plot.show()

razlicnih = np.asarray([np.unique(X[:,i]).size for i in xrange(n)])
plot.hist(razlicnih,bins=50,log=True)
plot.xlabel("stevilo razlicnih vrednosti")
plot.ylabel("st. atributov")
plot.show()


razlicnihProcent = (razlicnih-1+(nenicelnih==m))*100.0/nenicelnih
plot.hist(razlicnihProcent,bins=50,log=True)
plot.xlabel("stevilo razlicnih v odvisnosti od stevila nenicelnih")
plot.ylabel("st. atributov")
plot.show()




