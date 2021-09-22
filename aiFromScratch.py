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
plt.scatter(X[:,0], X[:,1])
plt.show()
