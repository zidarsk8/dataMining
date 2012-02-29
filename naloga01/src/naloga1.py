import matplotlib.pyplot as plot
import itertools
import collections
from sets import Set


td = open("trainingData.csv")
tl = open("trainingLabels.csv")


labels = [i.strip().split(",") for i in tl.readlines()]
primeri = [i.strip().split("\t") for i in td.readlines()]


print "Koliko primerov in atributov vsebujejo podatki? Kaksnega tipa so atributi?"
print "stevilo atributov %d" % len(primeri[0])
print "stevilo primerov  %d" % len(primeri)
print "----------------------\n"


#flatArr = reduce( (lambda x,y: x+y) , primeri)
td.seek(0)
flatArr = td.read().replace("\n","\t").strip().split("\t")
dataAll = len(flatArr)
dataZero = sum([i=="0" for i in flatArr])
dataNotZ = dataAll - dataZero
print "Kako redka je matrika oz. kaksen delz njenih elementov ima vrednost razlicno od 0?"
print "stevilo vseh elementov  %d" % dataAll
print "stevilo vseh nicelnih   %d" % dataZero
print "stevilo vseh nenicelnih %d" % dataNotZ
print "delez vseh nenicelnih   %f" % (float(dataNotZ)/dataAll)
print "----------------------\n"


nonZeroAttr = [sum([i != "0" for i in a]) for a in primeri]
print "Koliko atributov ima vrednost razlicno od 0 za posamezen primer?"
print "max stevilno nenicelnih atributov na primer %d" % max(nonZeroAttr)
print "min stevilno nenicelnih atributov na primer %d" % min(nonZeroAttr)
print "----------------------\n"
plot.hist(nonZeroAttr,bins=20)
plot.xlabel("st. nenicelnih atributov")
plot.ylabel("st. primerov")
plot.savefig("nonZeroAttr.pdf")
plot.close()


atributi = zip(*primeri)
nonZeroPrimeri = [sum([i != "0" for i in a]) for a in atributi]
print "V koliko primerih atribut zavzame nenicelne vrednosti?"
print "max stevilno nenicelnih primerov na atribut %d" % max(nonZeroPrimeri)
print "min stevilno nenicelnih primerov na atribut %d" % min(nonZeroPrimeri)
print "----------------------\n"
plot.hist(nonZeroPrimeri,bins=50,log=True)
plot.xlabel("st. nenicelnih vrednosti")
plot.ylabel("st. atributov")
plot.savefig("nonZeroPrimeri.pdf")
plot.close()


labelCounts = [len(i) for i in labels]
maxOznak = max(labelCounts)
print "Koliko je vseh razlicnih oznak (razredov) v podatkih?"
print "S koliko razlicnimi oznakami so oznaceni primeri?"
print "stevilo oznak (razredov) %d" % len(Set(itertools.chain(*labels)))
print "max razlicnih oznak      %d" % maxOznak
print "min razlicnih oznak      %d" % min(labelCounts)
print "----------------------\n"
plot.hist(labelCounts,bins=max(labelCounts))
plot.xlabel("stevilo oznak")
plot.ylabel("stevilo primerov")
plot.savefig("labelCountsMax.pdf")
plot.close()



counter=collections.Counter(nonZeroAttr)
print "Nastej 3 najbolj pogosta stevila nenicelnih vrednosti atributov pri posameznih primerih."
print counter.most_common(3)





maxAttr=[]
for i in range(len(nonZeroAttr)):
	if (labelCounts[i] == maxOznak):
		maxAttr.append(nonZeroAttr[i])

print "koliko atributov je takih ki imajo najvec oznak, in koliko atributov imajo taki primeri?"
print "stevilo primerov      %d" % len(maxAttr)
print "max stevilo atributov %d" % max(maxAttr)
print "min stevilo atributov %d" % min(maxAttr)
if len(maxAttr)>3: #manj kot toliko se nam ne splaca risati
	plot.hist(maxAttr)
	plot.xlabel("stevilo oznak")
	plot.ylabel("stevilo primerov")
	plot.savefig("maxAttr.pdf")
	plot.close()



counter=collections.Counter(labelCounts)
mostCommon = counter.most_common(1)[0][0]
commonAttr=[]
for i in range(len(nonZeroAttr)):
	if (labelCounts[i] == mostCommon):
		commonAttr.append(nonZeroAttr[i])
print "koliko atributov imajo primeri z najbolj pogostim stevilom oznak in koliko je takih primerov?"
print "stevilo primerov      %d" % len(commonAttr)
print "max stevilo atributov %d" % max(commonAttr)
print "min stevilo atributov %d" % min(commonAttr)
if len(commonAttr)>3:
	plot.hist(commonAttr)
	plot.xlabel("stevilo oznak")
	plot.ylabel("stevilo primerov")
	plot.savefig("commonAttr.pdf")
	plot.close()


