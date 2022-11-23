### Imports

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

import seaborn as sns

### Functions

def display(clf, data, label, title):
    plt.figure(figsize=(10,10))
    plt.suptitle(title)
    
    plt.subplot(1,2,1)
    plt.title("Training loss")
    plt.ylabel("Value of loss")
    plt.xlabel("Epochs")
    y = clf.loss_curve_
    x = list(range(1,len(y)+1))
    plt.plot(x,y)
    
    plt.subplot(1,2,2)
    plt.title("HeatMap")
    plt.ylabel("Predicted labels")
    plt.xlabel("Real labels")
    y_pred = clf.predict(data)
    cm = confusion_matrix(label_target, y_pred)
    ax = sns.heatmap(cm, center=True)
    ax.set(xlabel="Training labels", ylabel="Predicted labels")
    

    #plt.show()
    plt.savefig(f'./Desktop/IRD/Multilayer-perceptron/results/{title[:-33]}.png')
    plt.clf()


def train_model(layer, tols, epochs) :
    loo = LeaveOneOut()
    results = np.zeros(len(data))
    for train, test in loo.split(data):
        clf = MLPClassifier(hidden_layer_sizes=layer, random_state=1, max_iter=epochs, tol = tols).fit(data[train], label_target[train])
        results[test] = clf.predict(data[test])
        
    accuracy = (accuracy_score(label_target, results)*100)
    title = f"Results for {layer[1]} layers of {layer[0]} nodes, {epochs} epochs and a tol of {tols} \n Accuracy = {accuracy} %"
    display(clf, data, label_target, title)

def train_parameters() :
    layer = [(25,1),(50,1),(25,10),(50,10)]
    tols = [10e-4, 10e-5, 10e-3, 10e-2]
    epochs = [20,50,100]
    
    for i in layer :
        train_model(i, tols[0], epochs[0])
    for j in tols :
        train_model(layer[0], j, epochs[0])
    for k in epochs :
        train_model(layer[0], tols[0], k)

def search_best_parameter():
    
    layer = [(25,1),(50,1),(25,10),(50,10)]
    tols = [10e-4, 10e-5, 10e-3, 10e-2]
    epochs = [20,50,100]
    
    loo = LeaveOneOut()
    results = np.zeros(len(data))
    for train, test in loo.split(data):
        clf = GridSearchCV(
            MLPClassifier(),
            param_grid= {
                'hidden_layer_sizes': layer,
                'tol' : tols,
                'max_iter' : epochs
                },
            refit='True',
            cv=2,
            n_jobs=-1,
        )
        clf.fit(data[train], label_target[train])
        results[test] = clf.predict(data[test])
    print ("Accuracy=%f%%" % (accuracy_score(label_target, results)*100))
    return (clf.best_params_)

### Main

Data_Pam = pd.read_csv("./Desktop/IRD/Multilayer-perceptron/data/Data_Pam50.csv")

label_raw = Data_Pam.iloc[:,1]

data_raw = Data_Pam.iloc[:,2:]
data_raw_transposed = data_raw.T

values_tot = []
for j in data_raw_transposed :
    values = []
    for k in data_raw_transposed[j] :
        values.append(k)
    values_tot.append(values)
data = np.array(values_tot)

target = {'Normal': 0, 'Basal': 1, 'Her2': 2, 'LumA': 3, 'LumB': 4}
label_target = np.zeros(len(label_raw))
for i in range (len(label_raw)) :
    label_target[i] = target[label_raw[i]]

train_parameters()