import pandas as pd
from preprocessing import preprocess_data
from features import feature_selection
from modeling import modeling_step
from tuning import hyperparameter_tuning

def main():
    df = pd.read_csv('Breast_Cancer_dataset.csv')
    df_preprocessed = preprocess_data(df, n_components=4, options={"plot": True})
    X_train, X_test, y_train, y_test = feature_selection(df_preprocessed)
    modeling_step(X_train, X_test, y_train, y_test)
    hyperparameter_tuning(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()