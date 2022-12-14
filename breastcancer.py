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


# Plot a bar graph containing following values

# 1. Accuracy
# 2. Name

import  matplotlib.pyplot  as  plt
name = ["Logistic Regression", "Decision Tree", "Random Forest", "GradientBoostingClassifier"]
accuracy = [0.96, 0.95, 0.97, 0.98]

plt.bar(name, accuracy, color ='maroon', width = 0.4)





