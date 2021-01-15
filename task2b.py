import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from graphviz import Digraph
import graphviz
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import itertools
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#	Parse data
dfLife = pd.read_csv("life.csv", encoding = "ISO-8859-1")
dfWorld = pd.read_csv("world.csv", encoding = "ISO-8859-1")

#	Merge the dataframes, remove rows if they can't find a match
data = dfLife.merge(dfWorld, on="Country Code")
data = data.sort_values("Country Code")

#	Replace incomplete data with nan
data = data.replace("..", np.nan)
columnNames = list(data.columns[6:])
label = data["Life expectancy at birth (years)"]
data = data[columnNames]


#	Split data into a 70:30 split for training:test 
rState = 139
X_train, X_test, y_train, y_test = train_test_split(data,label,train_size=0.70,test_size=0.3,random_state=rState)

trainIndex = X_train.index
testIndex = X_test.index

#	Impute data to handle incomplete data
imp = SimpleImputer(missing_values = np.nan, strategy="median")
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

#	Normalise the data to have 0 mean and variance
scalar = preprocessing.StandardScaler().fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

#	Put data back into a dataframe
X_train = pd.DataFrame(X_train)
X_train.columns = columnNames
X_train.index = trainIndex
X_test = pd.DataFrame(X_test)
X_test.columns = columnNames
X_test.index = testIndex

#	Copy the dataframe to use on the other models
X_trainCopy = X_train.copy(deep=True)
X_testCopy = X_test.copy(deep=True)


#	Generate feature pairs
for i in range(0, 20):
	for j in range(i+1, 20):
		newName = columnNames[i] + " " + columnNames[j]
		X_train[newName] = X_train[columnNames[i]] * X_train[columnNames[j]]
		X_test[newName] = X_test[columnNames[i]] * X_test[columnNames[j]]

#	Graph that was used to decide the number of clusters
mms = MinMaxScaler()
mms.fit(X_trainCopy)
kData = mms.transform(X_trainCopy)
Sum_of_squared_distances = []
K = range(1,25)
for k in K:
    km = KMeans(n_clusters=k, random_state=rState)
    km = km.fit(kData)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, "bx-")
plt.xlabel("k value")
plt.ylabel("Sum_of_squared_distances")
plt.title("Elbow Method For Optimal k Value")
plt.savefig("task2bgraph1.png")

#	Clusters was determined by using elbow method
clusters = 5
KmeanTrain = KMeans(n_clusters=clusters, random_state=rState)
KmeanTrain.fit(X_trainCopy)
X_train["Kmean"] = KmeanTrain.predict(X_trainCopy)
KmeanTest = KMeans(n_clusters=clusters, random_state=rState)
KmeanTest.fit(X_testCopy)
X_test["Kmean"] = KmeanTest.predict(X_testCopy)

#	Find the top 4 correlated columns using mutual information algorithm
mutualInfo = mutual_info_classif(X_train, y_train, random_state=rState)
top4 = list(np.argsort(mutualInfo)[-4:])
featureEngTrain = X_train.iloc[:, top4]
featureEngTest = X_test.iloc[:, top4]

#	Feature engineering model
knn3Feature = neighbors.KNeighborsClassifier(n_neighbors=3)
knn3Feature.fit(featureEngTrain, y_train)
knn3FeatureAcc = accuracy_score(y_test, knn3Feature.predict(featureEngTest))
print(f"Accuracy of feature engineering: {knn3FeatureAcc:.3f}")

#	PCA model
pca = PCA(n_components=4)
pca.fit(X_trainCopy)
knn3PCA = neighbors.KNeighborsClassifier(n_neighbors=3)
knn3PCA.fit(pca.transform(X_trainCopy), y_train)
knn3PCAAcc = accuracy_score(y_test, knn3PCA.predict(pca.transform(X_testCopy)))
print(f"Accuracy of PCA: {knn3PCAAcc:.3f}")

#	First four features model
knn3First4 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn3First4.fit(X_trainCopy.iloc[:, 0:4], y_train)
knn3First4Acc = accuracy_score(y_test, knn3First4.predict(X_testCopy.iloc[:, 0:4]))
print(f"Accuracy of first four features: {knn3First4Acc:.3f}")