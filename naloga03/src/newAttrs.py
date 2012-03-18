import data

d = data.getDataArray()

a10 = data.getBadAttributes(d,10)
d10 = data.filterArr(d,a10) 

a20 = data.getBadAttributes(d,20)
d20 = data.filterArr(d,a20) 

a40 = data.getBadAttributes(d,40)
d40 = data.filterArr(d,a40) 

print len(d10[0])
print len(d20[0])
print len(d40[0])


