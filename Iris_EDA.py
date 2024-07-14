
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

iris = sns.load_dataset('iris')
iris

iris.info()

iris.describe()
iris[iris.duplicated()]
iris['species'].value_counts()
plt.title('Species Count')
sns.countplot(iris['species'])
plt.figure(figsize=(17,9))
plt.title('Comparison between various species based on sapel length and width')
sns.scatterplot(x=iris['sepal_length'], y=iris['sepal_width'], hue=iris['species'], s=50)

plt.figure(figsize=(16,9))
plt.title('Comparison between various species based on petal lenght and width')
sns.scatterplot(x=iris['sepal_length'], y=iris['sepal_width'], hue=iris['species'], s=50)
numeric_cols = iris.select_dtypes(include=['float64', 'int64'])
sns.pairplot(iris,hue="species",height=4)
plt.figure(figsize=(10,11))
sns.heatmap(numeric_cols.corr(), annot=True)
plt.plot()

fig, axes = plt.subplots(2, 2, figsize=(16,9))
sns.boxplot( y="petal_width", x= "species", data=iris, orient='v' , ax=axes[0, 0])
sns.boxplot( y="petal_length", x= "species", data=iris, orient='v' , ax=axes[0, 1])
sns.boxplot( y="sepal_length", x= "species", data=iris, orient='v' , ax=axes[1, 0])
sns.boxplot( y="sepal_width", x= "species", data=iris, orient='v' , ax=axes[1, 1])
plt.show()
sns.FacetGrid(iris, hue="species", height=5) \
.map(sns.histplot, "sepal_length") \
.add_legend()

sns.FacetGrid(iris, hue="species", height=5) \
.map(sns.histplot, "sepal_width") \
.add_legend()

sns.FacetGrid(iris, hue="species", height=5) \
.map(sns.histplot, "petal_length") \
.add_legend()

sns.FacetGrid(iris, hue="species", height=5) \
.map(sns.histplot, "petal_width") \
.add_legend()
plt.show()



