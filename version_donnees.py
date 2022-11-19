### IMPORTS

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

### Read csv file

Data_Pam = pd.read_csv("./Desktop/IRD/Multilayer-perceptron/data/Data_Pam50.csv")

label_raw = Data_Pam.iloc[:,1]

data_raw = Data_Pam.iloc[:,2:]
data_raw_transposed = data_raw.T

### Visualisation of raw data

''' A tester au CREMI
df = Data_Pam.iloc[:,2:]
features = list(df)
df = Data_Pam.iloc[:,1:]

scatter_matrix(df[features], c=df['subtype'].apply(lambda subtype: {
        'Normal': 'red',
        'Basal': 'pink',
        'Her2': 'orange',
        'LumA': 'blue',
        'LumB': 'green'
    }[subtype]),
     alpha=0.8, figsize=(12, 12)
)
'''

### Creation of array containing data and labels

lab = []
for i in label_raw :
    lab.append(i)
label = np.array(lab)

values_tot = []
for j in data_raw_transposed :
    values = []
    for k in data_raw_transposed[j] :
        values.append(k)
    values_tot.append(values)
data = np.array(values_tot)

### Use of MLPClassifier for with a part of origin data as train, the other part as test

'''
Data_train, Data_test, label_train, label_test = train_test_split(data, label, stratify=label, random_state=1)

clf = MLPClassifier(random_state=1, max_iter=300).fit(Data_train, label_train)

proba = clf.predict_proba(Data_test[:1])
predict = clf.predict(Data_test[:5, :])
score = clf.score(Data_test, label_test)
'''

### Use of LeaveOneOut, use each data as test only one time and train with others, allows to give accuracy for short data range

target = {'Normal': 0, 'Basal': 1, 'Her2': 2, 'LumA': 3, 'LumB': 4}
label_target = np.zeros(len(label))
for i in range (len(label)) :
    label_target[i] = target[label[i]]

def train_model(param) :
    loo = LeaveOneOut()
    results = np.zeros(len(data))
    for train, test in loo.split(data):
        clf = MLPClassifier(random_state=1, max_iter=300).fit(data[train], label_target[train])
        results[test] = clf.predict(data[test])
    print ("Accuracy=%f%%" % (accuracy_score(label_target, results)*100))

### Visualisation data and MLPClassifier

Data_train, Data_test, label_train, label_test = train_test_split(data, label_target, stratify=label, random_state=1)
figure = plt.figure(figsize=(17, 9))
i = 1
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#00FF00", "#FCFF33", "#FF33F9", "#0000FF"])
ax = plt.subplot(1,2,1)

ax.scatter(Data_train[:, 0], Data_train[:, 1], c=label_train, cmap=cm_bright)

ax.scatter(Data_test[:, 0], Data_test[:, 1], c=label_test, cmap=cm_bright, alpha=0.6)
plt.show()