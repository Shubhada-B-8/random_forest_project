from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
tree.plot_tree(rf_model.estimators_[0], 
               feature_names=iris.feature_names, 
               class_names=iris.target_names,
               filled=True, 
               rounded=True, 
               fontsize=10)

plt.title("Decision Tree from Random Forest")
plt.show()
