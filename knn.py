import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Loading in the data

data = pd.read_csv(os.getcwd() + '/data/creditcard-fraud-dataset.csv')
df = pd.DataFrame(data)

X = df.iloc[:, : -1]
y = df.iloc[:, -1]

# Defining the test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

scaler = StandardScaler()
labeler = LabelEncoder()
scaler.fit_transform(X_train)
labeler.fit_transform(y_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = labeler.transform(y_train)
y_test = labeler.transform(y_test)


# Finding the optimal KNN estimator
grid_params = {
        'n_neighbors': [1, 3, 5, 7, 9 , 11],
        'weights': ['uniform', 'distance'],
        'metric': ['chebyshev', 'minkowski']
        }
KNN_GV = GridSearchCV(KNeighborsClassifier(), grid_params, cv = 5)
KNN_GV.fit(X_train, y_train)


# Printing out the results
print(f'The best parameters are {KNN_GV.best_params_}')
print(f'The best accuracy on the training data is {KNN_GV.score(X_train, y_train)}')
print(f'The best accuracy on the testing data is {KNN_GV.score(X_test, y_test)}')
