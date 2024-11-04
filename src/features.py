import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split

def feature_selection(df):
    X = df.drop('Status', axis=1)
    y = df['Status'].apply(lambda x: 1 if x == 'Alive' else 0) 

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)
    
    # Feature Selection and Ranking using Mutual Information
    selector = SelectKBest(mutual_info_classif, k=10)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    print("Selected Features based on Mutual Information:")
    print("\n".join(list(selected_features)))
    
    X_selected = X[selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    y_train = y_train.values
    y_test = y_test.values

    return X_train, X_test, y_train, y_test
