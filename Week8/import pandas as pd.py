import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("C:\\Users\\Hp\\OneDrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\Week6\\drug200.csv")

print(df)

print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.head())
print(df.tail())

X = df[["Age","Sex","BP","Cholesterol","Na_to_K"]]
y = df[["Drug"]]

X.loc[:,"Sex"]=X["Sex"].map({"F": 0,"M": 1})
X.loc[:,"BP"]=X["BP"].map({"High":0,"Low":1})
X.loc[:,"Cholesterol"]=X["Cholesterol"].map({"High":0,"Normal":1})

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object

model = DecisionTreeClassifier()
# Train Decision Tree Classifer
model.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, filled=True)
plt.show()

