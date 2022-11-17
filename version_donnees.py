import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


Data_Pam = pd.read_csv("./Desktop/IRD/Multilayer-perceptron/data/Data_Pam50.csv")

label_raw = Data_Pam.iloc[:,1]

data_raw = Data_Pam.iloc[:,2:]
data_raw_transposed = data_raw.T

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

Data_train, Data_test, label_train, label_test = train_test_split(data, label, stratify=label, random_state=1)

clf = MLPClassifier(random_state=1, max_iter=300).fit(Data_train, label_train)

proba = clf.predict_proba(Data_test[:1])
predict = clf.predict(Data_test[:5, :])
score = clf.score(Data_test, label_test)
