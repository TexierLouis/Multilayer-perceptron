'''
Doc for concept : 
https://scikit-learn.org/stable/modules/neural_networks_supervised.html
Doc for MLPClassifier : 
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
Doc for MLPRegressor :
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

'''

##### MLPClassifier #####

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate cluster of data
X, y = make_classification(n_samples=100, random_state=1)

# Split data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

# Multi-layer Perceptron classifier
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

# Parameters and output
clf.predict_proba(X_test[:1])
clf.predict(X_test[:5, :])
clf.score(X_test, y_test)

##### MLPRegressor #####

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

# Generate regressive data
Z, t = make_regression(n_samples=100, random_state=1)

# Split data into training data and testing data
Z_train, Z_test, t_train, t_test = train_test_split(Z, t, random_state=1)

# Multi-layer Perceptron regressor
regr = MLPRegressor(random_state=1, max_iter=300).fit(Z_train, t_train)

# Parameters and output
regr.predict(Z_test[:2])
regr.score(Z_test, t_test)