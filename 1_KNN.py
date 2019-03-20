# K-Nearest Neighbours
# classifies data point => neighbours are classified

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
# print(dir(iris))
# print(iris['data'][0])
# print(iris['feature_names'])

# ======================
# create dataframe

dfIris = pd.DataFrame(
    iris['data'],
    columns = ['sepalL', 'sepalW', 'petalL', 'petalW']
)
dfIris['target'] = iris['target']
dfIris['spesies'] = dfIris['target'].apply(
    lambda e: iris['target_names'][e]
)
# print(dfIris.head(3))

# df0 , df1 , df2
df0 = dfIris[dfIris['spesies'] == 'setosa']
df1 = dfIris[dfIris['spesies'] == 'versicolor']
df2 = dfIris[dfIris['spesies'] == 'virginica']

# plot dataframe df0, df1, df2
plt.scatter(
    df0['sepalL'],
    df0['sepalW'],
    marker = 'o',
    color = 'r'
)
plt.scatter(
    df1['sepalL'],
    df1['sepalW'],
    marker = 'o',
    color = 'y'
)
plt.scatter(
    df2['sepalL'],
    df2['sepalW'],
    marker = 'o',
    color = 'b'
)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.grid(True)
# plt.show()

# ===========================
from sklearn.model_selection import train_test_split
a,b,c,d = train_test_split(
    dfIris[['sepalL', 'sepalW', 'petalL', 'petalW']],
    dfIris['target'],
    test_size = .1
)
# print(a)
# print(b)

# =========================
# KNN

# 1. sqrt(n_data)
# 2. odd
# k = round((len(a)+len(b)) ** .5)
# print(k)

def nilai_k():
    k = round((len(a)+len(b)) ** .5)
    if (k % 2 == 0):
        return k + 1
    else:
        return k
print(nilai_k())

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(
    n_neighbors = nilai_k()
)

# training
model.fit(a, c)

# accuracy
print(round(model.score(b, d) * 100), '%')

# predict

print(model.predict([[4.5,2.3, 1.3, 0.3]]))
print(df0['target'].iloc[41])

# print(b.iloc[0])
# print(model.predict([b.iloc[0]]))
# print(d.iloc[0])