#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:49:44 2020

@author: luismiguel
"""

import pandas as pd

data = pd.read_csv('/Users/luismiguel/Downloads/US_Accidents_Dec19.csv')

df = data.sample(n=500000, random_state = 5)
list_topics = [3,4,5,6,7,8,9,10,11,13,14,15,16,17,20,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
df = df.iloc[:,list_topics]

df.to_csv('/Users/luismiguel/Desktop/McGill MMA/Enterprise Analytics/accidents.csv')

### only numerical values

list_numerical = [0, 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
list_drop = [0,1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
#intuition = [5,6,7,10, 17]
weather = [0,1,2,3,4,5,6,7]
df = df.iloc[:, list_numerical]
df = df.iloc[:, list_drop]
#df = df.iloc[:, intuition]
df = df.iloc[:, weather]


df["Sunrise_Sunset"]  = [1 if x == 'Day' else 0 for x in df['Sunrise_Sunset']]
df["Civil_Twilight"]  = [1 if x == 'Day' else 0 for x in df['Civil_Twilight']]
df["Nautical_Twilight"]  = [1 if x == 'Day' else 0 for x in df['Nautical_Twilight']]
df["Astronomical_Twilight"]  = [1 if x == 'Day' else 0 for x in df['Astronomical_Twilight']]

df = df.fillna(value=0)
#######   STANDARDIZE   ########

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(df)


###### K-MEANS #########

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
model= kmeans.fit(X_std)
labels = model.predict(X_std)

    

from matplotlib import pyplot
pyplot.scatter(df['Precipitation(in)'], df['Visibility(mi)'], c=labels, cmap='rainbow')  


df['cluster'] = labels
cluster_mean = df.groupby('cluster').mean()



#3D CLUSTER
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = ax = Axes3D(fig)

x = df['Visibility(mi)']
y = df['Precipitation(in)']
z = df['Severity']
ax.scatter(x,y,z, c=labels, cmap='rainbow')
ax.set_xlabel('Visibility')
ax.set_ylabel('Precipitation')
ax.set_zlabel('Severity')
pyplot.show()



withinss = []
### type your code here
for i in range(2,8):
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X_std)
    withinss.append(model.inertia_)
    
from matplotlib import pyplot
### type your code here
pyplot.plot([2,3,4,5,6,7],withinss)

# cmap='rainbow', depthshade=True
######### PCA ###########

from sklearn.decomposition import PCA
X_std_pca = pd.DataFrame(X_std)
variables = [0,5]
X_std_pca = X_std_pca.iloc[:, variables]
X_std_pca = scaler.fit_transform(X_std_pca)
pca = PCA(n_components=2)
pca.fit(X_std_pca)
print(pca.components_)
print(pca.explained_variance_)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0, color='red')
    ax.annotate('', v1, v0, arrowprops=arrowprops)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

plt.scatter(X_std_pca[:, 0], X_std_pca[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 30 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)   
plt.axis('equal');