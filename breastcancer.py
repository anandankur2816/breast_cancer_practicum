import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

df = pd.read_csv("data.csv")
df.drop(df.columns[[0,-1]], axis=1, inplace=True)

# Split the features data and the target 
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# Encoding the target value 
y = np.asarray([1 if c == 'M' else 0 for c in y])
cols = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean']
X = df[cols]
print(X.columns)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=43)
print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

clf_report = classification_report(y_test, y_pred)
print('Classification report')
print("---------------------")
print(clf_report)
print("_____________________")

# joblib.dump(rfc,"cancer_model.pkl")

# Plot the dependency of the accuracy on the number of trees
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
param_range = np.arange(1, 250, 2)
train_scores, test_scores = validation_curve(
    RandomForestClassifier(), X, y, param_name="n_estimators", param_range=param_range,
    cv=3, scoring="accuracy", n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.title("Validation Curve with Random Forest")
plt.xlabel("Number of trees")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_mean, label="Training score",
                color="darkorange", lw=lw)
plt.fill_between(param_range, train_mean - train_std,
                train_mean + train_std, alpha=0.2,
                color="darkorange", lw=lw)
plt.plot(param_range, test_mean, label="Cross-validation score",
                color="navy", lw=lw)
plt.fill_between(param_range, test_mean - test_std,
                test_mean + test_std, alpha=0.2,
                color="navy", lw=lw)
plt.legend(loc="best")
plt.show()




