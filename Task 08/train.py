from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

#Load dataset
digits = load_digits()
X, y = digits.data, digits.target 

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

#Train model
knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(X_train, y_train)

#Accuracy
print("Accuracy: ", knn.score(X_test, y_test))

#Create 'model' directory if it doesnt exist
os.makedirs('model', exist_ok= True)

#Save model
joblib.dump(knn, "model/knn_model.pkl")