import matplotlib.pyplot as plt
import numpy as np 
import scipy as sc
from sklearn.datasets import make_circles

#make datasets
samples = 500
features = 2  #bidimensional

#plot
X, Y = make_circles(n_samples = samples,factor=0.5,noise=0.06)
print(Y)
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c= '#6a040f')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c= '#0a9396')
plt.axis("equal")
plt.show()
