import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def GridSearch(estimator, param_grid, X_train, X_test, y_train, y_test, cv=5, scoring='roc_auc'):
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,
                               n_jobs=-1, scoring=scoring, verbose=1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best = grid_search.best_estimator_
    y_pred = best.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_prob = best.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    return y_pred, accuracy, best_params, roc_auc

def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    # Stratified K-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -----------------------------
    # Hyperparameter Tuning for Gradient Boosting
    # -----------------------------

    print("\nStarting Hyperparameter Tuning for Gradient Boosting...")

    param_grid_gb = {
        'n_estimators': [50, 75, 100],      
        'learning_rate': [0.1, 0.15, 0.2],  
        'max_depth':  [2, 3],                  
        'min_samples_split': [2, 3],          
        'min_samples_leaf': [5, 6, 8, 10],           
        'max_features': ['sqrt'],                      
    }

    gb = GradientBoostingClassifier(random_state=42)
    y_pred_gb, accuracy_gb, best_params_gb, roc_auc_gb = GridSearch(
        gb, param_grid_gb, X_train, X_test, y_train, y_test, cv=cv, scoring='roc_auc')

    print("\nBest Hyperparameters for Gradient Boosting:")
    print(best_params_gb)
    print(f"Gradient Boosting Accuracy with Best Hyperparameters: {accuracy_gb:.4f}")
    print(f"Gradient Boosting ROC AUC Score with Best Hyperparameters: {roc_auc_gb:.4f}")

    print("\nClassification Report for Gradient Boosting:")
    print(classification_report(y_test, y_pred_gb))

    return

    # -----------------------------
    # Hyperparameter Tuning for Neural Network
    # -----------------------------

    print("\nStarting Hyperparameter Tuning for Neural Network...")
    param_grid_nn = {
        'hidden_layer_sizes': [(50,),(100,),(50,50),(100,100)],  
        'activation': ['relu'],                         
        'solver': ['adam'],                                                    
        'learning_rate': ['adaptive'],               
        'learning_rate_init': [0.02],                     
        'max_iter': [200]                                  
    }

    nn = MLPClassifier(random_state=42)
    y_pred_nn, accuracy_nn, best_params_nn, roc_auc_nn = GridSearch(
        nn, param_grid_nn, X_train, X_test, y_train, y_test, cv=cv, scoring='roc_auc')

    print("\nBest Hyperparameters for Neural Network:")
    print(best_params_nn)
    print(f"Neural Network Accuracy with Best Hyperparameters: {accuracy_nn:.4f}")
    print(f"Neural Network ROC AUC Score with Best Hyperparameters: {roc_auc_nn:.4f}")

    print("\nClassification Report for Neural Network:")
    print(classification_report(y_test, y_pred_nn))

    print("\nComparison of Models after Hyperparameter Tuning:")
    results = pd.DataFrame({
        'Model': ['Gradient Boosting', 'Neural Network'],
        'Accuracy': [accuracy_gb, accuracy_nn],
        'ROC AUC': [roc_auc_gb, roc_auc_nn]
    })
    print(results)
