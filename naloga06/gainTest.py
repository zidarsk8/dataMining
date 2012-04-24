
# turns list of true and false strings into an int TTFT = 1101 = 13 
def listToIntTF(l,t,f):
	t = str(t)
	f = str(f)
	i = int("".join(l).replace(t,"1").replace(f,"0"),2)
	return {"n":i, "l":len(l), "c":countOnes(i)}

# calculates information gain between two bitmaps (integers)
# x,y must be {n: number, c: numberOfOnes, s: sizeRepresented}
# I should get rid of the size since it's in x and y but too lazy
def gain(x,y,size):
	x1 = x["c"]
	y1 = y["c"]
	x1y1 = countOnes(x["n"]&y["n"])
	x1y0 = x1 - x1y1
	x0y1 = y1 - x1y1
	x0y0 = size - x1y1 - x1y0 - x0y1
	px1 = x1 / float(size)
	py1 = y1 / float(size)
	px0 = 1 - px1
	py0 = 1 - py1
	px1y1 = x1y1 / float(size)
	px1y0 = x1y0 / float(size)
	px0y1 = x0y1 / float(size)
	px0y0 = x0y0 / float(size)
	return  (0 if x1y1==0 else px1y1 * log((px1y1)/(px1*py1),2)) + \
		(0 if x0y1==0 else px0y1 * log((px0y1)/(px0*py1),2)) + \
		(0 if x1y0==0 else px1y0 * log((px1y0)/(px1*py0),2)) + \
		(0 if x0y0==0 else px0y0 * log((px0y0)/(px0*py0),2)) 

