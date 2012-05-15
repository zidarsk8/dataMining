import matplotlib.pyplot as plt
import pickle
import numpy as np

ll = pickle.load(file("rf_depth_10_400_data_1000_400.pkl"))

ll[""]

data = np.array(ll["aa"])
x = data[:,0]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, data[:,1], '-')
ax.plot(x, data[:,2], '-')
ax.plot(x, data[:,3], '-')
fig.autofmt_xdate()

plt.show()

