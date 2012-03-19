import data
import math

d = data.getDataArray()
t = data.getTestArray()

a10 = data.getBadAttributes(d,10)
d10 = data.filterArr(d,a10) 
t10 = data.filterArr(t,a10)

binD = [[int(x>0) for x in i] for i in d10]
binT = [[int(x>0) for x in i] for i in t10]

logD = [[int(math.ceil(math.log(x) if x > 0 else 0)) for x in i] for i in d10]
logT = [[int(math.ceil(math.log(x) if x > 0 else 0)) for x in i] for i in t10]

newD = []
newT = []

for i in range(len(t10)):
	newD.append(list(d10[i])+list(binD[i])+list(logD[i]))
	newT.append(list(t10[i])+list(binT[i])+list(logT[i]))


f = file("plusBinLogTraingingData.csv","w")
f.write("\n".join(["\t".join([str(x).replace("c","") for x in i]) for i in newD ]))
f.flush()
f.close()
f = file("plusBinLogTestData.csv","w")
f.write("\n".join(["\t".join([str(x).replace("c","") for x in i]) for i in newT ]))
f.flush()
f.close()

print len(d10[0])
print len(d20[0])
print len(d40[0])


