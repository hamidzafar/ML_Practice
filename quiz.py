import numpy as np

x=np.array([7921, 5184, 8826, 4761])

print (x - np.mean(x)) / (1.0 * (max(x)-min(x)))

print np.shape(np.ones(2))