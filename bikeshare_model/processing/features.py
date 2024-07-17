import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variables: list):
        # YOUR CODE HERE
        if not isinstance(variables, list):
            raise ValueError("variable should be list of strings")

        if not all(isinstance(var, str) for var in variables):
            raise ValueError("all variables should be strings")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        weekday, dteday = self.variables
        self.fill_value = X[X[weekday].isnull() | (X[weekday] == '')].index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        weekday, dteday = self.variables
        X = X.copy()
        self.fill_value = X[X[weekday].isnull() | (X[weekday] == '')].index
        X.loc[self.fill_value, weekday] = X.loc[self.fill_value, dteday].dt.day_name().apply(lambda x: x[:3])
        print('Weekday imputed')
        return X
    
class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.fill_value=X[self.variables].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variables] = X[self.variables].replace('', pd.NA)
        X[self.variables]=X[self.variables].fillna(self.fill_value)
        print('Weathersit imputed')
        return X
    
class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        if not isinstance(mappings, dict):
            raise ValueError("mappings should be a dict")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        # Match the type of keys to the type of values in the DataFrame column
        self.mappings = {str(k) if X[self.variables].dtype == 'object' else int(k): v for k, v in self.mappings.items()}
        X[self.variables] = X[self.variables].map(self.mappings).astype('int')
        print(self.variables, 'mapped!')
        return X
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables: list, lower_quantile: float = 0.25, upper_quantile: float = 0.75):
        # YOUR CODE HERE
        if not isinstance(variables, list):
            raise ValueError("variable should be list of strings")

        if not all(isinstance(var, str) for var in variables):
            raise ValueError("all variables should be strings")

        if not (0 <= lower_quantile <= 1 and 0 <= upper_quantile <= 1):
            raise ValueError("quantiles should be between 0 and 1")

        self.variables = variables
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.bounds_ = {}
        for var in self.variables:
            lower_bound = X[var].quantile(self.lower_quantile)
            upper_bound = X[var].quantile(self.upper_quantile)
            self.bounds_[var] = (lower_bound, upper_bound)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        for var in self.variables:
            lower_bound, upper_bound = self.bounds_[var]
            X[var] = np.where(X[var] > upper_bound, upper_bound, X[var])
            X[var] = np.where(X[var] < lower_bound, lower_bound, X[var])
        print('Outlier removed')
        return X
    
# Define a function to drop specified columns
def drop_columns(X, col_to_drop):
    print("Dropping columns")
    return X.drop(columns=col_to_drop)

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode specified categorical columns """

    def __init__(self, variables: list):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.encoder.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        one_hot_encoded = self.encoder.transform(X[self.variables])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=self.encoder.get_feature_names_out(self.variables), index=X.index)
        X = X.drop(columns=self.variables)
        X = pd.concat([X, one_hot_encoded_df], axis=1)
        print(f'{self.variables} onehot encoded')
        return X