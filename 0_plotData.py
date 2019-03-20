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
plt.show()