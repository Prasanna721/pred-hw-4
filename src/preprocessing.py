import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def preprocess_data(df, n_components=5, options=None):
    if options is None:
        options = {}
    plot = options.get("plot", True)
    
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    def detect_outliers_iqr(data, num_cols):
        outlier_indices = []
        for col in num_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            outliers_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step)].index
            outlier_indices.extend(outliers_col)
        outlier_indices = list(set(outlier_indices))
        return outlier_indices

    outlier_indices = detect_outliers_iqr(df, num_cols)
    df = df.drop(outlier_indices).reset_index(drop=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    n_features = len(num_cols)
    if n_components > n_features:
        print(f"n_components ({n_components}) is greater than the number of numerical features ({n_features}). Setting n_components to {n_features}.")
        n_components = n_features

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    df_svd = svd.fit_transform(df[num_cols])
    df_svd = pd.DataFrame(df_svd, columns=[f'SVD_{i+1}' for i in range(n_components)])
    df_final = pd.concat([df_svd.reset_index(drop=True), df[categorical_cols].reset_index(drop=True)], axis=1)

    if plot:
        print(svd.explained_variance_ratio_)
        explained_variance = svd.explained_variance_ratio_.cumsum()
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, n_components+1), explained_variance, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('SVD Explained Variance')
        plt.grid(True)
        plt.show()

    return df_final
