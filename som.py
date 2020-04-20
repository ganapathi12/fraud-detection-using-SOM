# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
sc.fit_transform(X)

#training the som
from minisom import MiniSom
som=MiniSom(10, 10, 15)
som.random_weights_init(X)
som.train_random(X, 100)

#visualise the results
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],markeredgecolor=colors[y[i]],markerfacecolor='None',markersize=10,markeredgewidth=2)
    
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,4)], mappings[(7,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)