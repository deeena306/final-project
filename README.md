# final-project

My Code Shows a Simple Clustering Technique
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
### Load the Iris Dataset
iris = datasets.load_iris()
df_iris = pd.DataFrame(data= np.c_[iris['data'], 
                                   iris['target']],
                     columns= iris['feature_names'] + ['target']
                      )

species = {0:'setosa',
           1:'versicolor',
           2:'virginica'}
X = df_iris.iloc[:, [0, 1, 2, 3]].values
df_iris['target'].replace(species, inplace=True)
df_iris.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
 4   target             150 non-null    object 
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
df_iris.head(1)
sepal length (cm)	sepal width (cm)	petal length (cm)	petal width (cm)	target
0	5.1	3.5	1.4	0.2	setosa
# Frequency Distribution of Species 
iris_outcome = pd.crosstab(index=iris["target"],  # Make a crosstab
                              columns="count")      # Name the count column

iris_outcome
col_0	count
row_0	
0	50
1	50
2	50
iris_setosa=df_iris.loc[df_iris["target"]=="Iris-setosa"]
iris_virginica=df_iris.loc[df_iris["target"]=="Iris-virginica"]
iris_versicolor=df_iris.loc[df_iris["target"]=="Iris-versicolor"]
sns.FacetGrid(df_iris,hue="target",height=3).map(sns.histplot,"sepal length (cm)").add_legend()
sns.FacetGrid(df_iris,hue="target",height=3).map(sns.histplot,"sepal width (cm)").add_legend()
sns.FacetGrid(df_iris,hue="target",height=3).map(sns.histplot,"petal length (cm)").add_legend()
plt.show()



sns.boxplot(x="target",y="petal length (cm)",data=df_iris)
plt.show()

sns.violinplot(x="target",y="petal length (cm)",data=df_iris)
plt.show()

Scatter Plot
sns.set_style("whitegrid")
sns.pairplot(df_iris,hue="target",size=3);
plt.show()
/opt/conda/envs/Python-3.9/lib/python3.9/site-packages/seaborn/axisgrid.py:2076: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
  warnings.warn(msg, UserWarning)

How to Implement KMeans
Choose the number of clusters k
Select K random points from the data as centroids
Assign all the points to the closest cluster centroid
Recompute the centroids of newly formed clusters
Repeat steps 3 and 4
#Finding the optimum number of clusters for k-means classification
wcss = []

X = df_iris[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df_iris['target']

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, 
                    init = 'k-means++', 
                    max_iter = 300, 
                    n_init = 10, 
                    random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
Using the elbox method to determine the optimal number of clusters for k-means clustering
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

# Number of Clusters Based on Elbow Analysis 
n_clusters = 3 
kmeans = KMeans(n_clusters = n_clusters, 
                init = 'k-means++', 
                max_iter = 300, 
                n_init = 10, 
                random_state = 0)
y_kmeans = kmeans.fit_predict(X)
X
sepal length (cm)	sepal width (cm)	petal length (cm)	petal width (cm)
0	5.1	3.5	1.4	0.2
1	4.9	3.0	1.4	0.2
2	4.7	3.2	1.3	0.2
3	4.6	3.1	1.5	0.2
4	5.0	3.6	1.4	0.2
...	...	...	...	...
145	6.7	3.0	5.2	2.3
146	6.3	2.5	5.0	1.9
147	6.5	3.0	5.2	2.0
148	6.2	3.4	5.4	2.3
149	5.9	3.0	5.1	1.8
150 rows × 4 columns

#Visualising the clusters
plt.scatter(X.loc[y_kmeans == 0, 'sepal length (cm)'], 
            X.loc[y_kmeans == 0, 'sepal width (cm)'], 
            s = 100, 
            c = 'purple', 
            label = 'setosa')
plt.scatter(X.loc[y_kmeans == 1, 'sepal length (cm)'], X.loc[y_kmeans == 1, 'sepal width (cm)'], s = 100, c = 'orange', label = 'versicolor')
plt.scatter(X.loc[y_kmeans == 2, 'sepal length (cm)'], X.loc[y_kmeans == 2, 'sepal width (cm)'], s = 100, c = 'green', label = 'virginica')

# #Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.legend()
<matplotlib.legend.Legend at 0x7ff69bb95220>

In this notebook we have covered the following: 1.
