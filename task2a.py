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
import csv

def writeCSV(data):
	with open("task2a.csv", "w", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(["feature", "median", "mean", "variance"])
		for i in range(0, len(data[0])):
			writer.writerow([data[0][i], data[1][i], data[2][i], data[3][i]])

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
X_train, X_test, y_train, y_test = train_test_split(data,label,train_size=0.70,test_size=0.3,random_state=200)

#	Need to grab median before x_train changes
median = X_train.median(axis=0, skipna=True)

#	Impute data to handle incomplete data
imp = SimpleImputer(missing_values = np.nan, strategy="median")
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

#	Normalise the data to have 0 mean and variance
scalar = preprocessing.StandardScaler().fit(X_train)
scalarMean = scalar.mean_
scalarVariance = scalar.var_
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

#	Create models
knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
knn7 = neighbors.KNeighborsClassifier(n_neighbors=7)
knn7.fit(X_train, y_train)
dTree = DecisionTreeClassifier(random_state=200,max_depth=3)
dTree.fit(X_train, y_train)
knn3Acc = accuracy_score(y_test, knn3.predict(X_test))
knn7Acc = accuracy_score(y_test, knn7.predict(X_test))
dTreeAcc = accuracy_score(y_test, dTree.predict(X_test))

print(f"Accuracy of decision tree: {dTreeAcc:.3f}")
print(f"Accuracy of k-nn (k=3): {knn3Acc:.3f}")
print(f"Accuracy of k-nn (k=7): {knn7Acc:.3f}")

writeCSV([columnNames, median, scalarMean, scalarVariance])