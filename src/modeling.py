import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def modeling_step(X_train, X_test, y_train, y_test):
    #K-Nearest Neighbors (KNN)
    class KNNClassifier:
        def __init__(self, k=5):
            self.k = k
        
        def fit(self, X, y):
            self.X_train = X.values.astype(float)
            self.y_train = y.astype(int)
        
        def predict(self, X):
            X = X.values.astype(float)
            y_pred = []
            for i in range(len(X)):
                distances = np.sqrt(np.sum((self.X_train - X[i])**2, axis=1))
                neighbor_indices = np.argsort(distances)[:self.k]
                neighbor_labels = self.y_train[neighbor_indices]
                counts = np.bincount(neighbor_labels)
                y_pred.append(np.argmax(counts))
            return y_pred
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    knn = KNNClassifier(k=5)
    knn.fit(X_train.reset_index(drop=True), y_train)
    y_pred_knn = knn.predict(X_test.reset_index(drop=True))
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"\nKNN Accuracy: {accuracy_knn:.4f}")
    
    #Naïve Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naïve Bayes Accuracy: {accuracy_nb:.4f}")
    
    #C4.5 Decision Tree 
    dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print(f"C4.5 Decision Tree Accuracy: {accuracy_dt:.4f}")
    
    #Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
    
    #Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    print(f"Gradient Boosting Accuracy: {accuracy_gb:.4f}")
    
    #Neural Network
    nn = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200, random_state=42)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    print(f"Neural Network Accuracy: {accuracy_nn:.4f}")
    
    results = pd.DataFrame({
        'Model': ['KNN', 'Naïve Bayes', 'C4.5 Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network'],
        'Accuracy': [accuracy_knn, accuracy_nb, accuracy_dt, accuracy_rf, accuracy_gb, accuracy_nn]
    })
    print("\nModel Performance:")
    print(results)
