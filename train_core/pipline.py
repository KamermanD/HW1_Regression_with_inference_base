
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

class MissingHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.medians = {}
        for col in self.columns:
            if col in X.columns:
                self.medians[col] = X[col].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].apply(lambda x: self.medians[col] if pd.isna(x) else x)
        return X

def convert_to_numeric(X):
    X = X.copy()
    for col in ['mileage', 'engine', 'max_power']:
        if col in X.columns:
            X[col] = X[col].str.extract(r'(\d+\.?\d*)').astype(float)
    return X

def remove_anomalies(X, y, thresholds):
    X = X.copy()
    y = y.copy()
    for col, threshold in thresholds.items():
        if col in X.columns:
            if col == 'mileage' or col == 'max_power':
                mask = X[col] >= threshold
            elif col == 'km_driven':
                mask = X[col] <= threshold
            X = X[mask]
            y = y[mask]
    return X.reset_index(drop=True), y.reset_index(drop=True)

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(self.columns, axis=1)

class NameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
     
        X['stamp'] = X['name'].apply(lambda x: x.split()[0])
        return X.drop('name', axis=1) 

    
numeric_features = ['mileage', 'engine', 'max_power', 'km_driven']
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('drop_torque', ColumnDropper(columns=['torque'])),
    ('missing_value_handler', MissingHandler(columns=['mileage', 'engine', 'max_power', 'seats'])),
    ('process_name', NameTransformer()),  
    ('preprocessor', preprocessor),  
    ('regressor', LinearRegression()) 
])
