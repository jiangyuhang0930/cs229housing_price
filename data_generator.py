import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataGenerator:
    """ Generates and cleans dataframe from csv files; encodes categorical inputs

    Args:
    data_path: path to the dataset file
    write: If True, calls write_to_csv to generated a csv containing the processed dataset.
    """

    def __init__(self, data_path, write = False):
        self.data_path = data_path
        self.X = pd.read_csv(data_path + '.csv', sep=',')
        target = 'Price' if 'melb' in data_path else 'price'
        self.seperate_feature_target(target)
        self.fillin_missing_values()
        self.str_to_categorical()
        self.scale()
        if write:
            self.write_to_csv()

    def seperate_feature_target(self, target_name):
        """ Separator features from targets
        Returns nothing, only update self.X and self.y
        """
        self.y = self.X[target_name]
        self.X = self.X.drop(columns=[target_name])

    def fillin_missing_values(self):
        X_means = self.X.mean().round(0)    
        X_modes = self.X.mode()
        # some columns are numeric but we want to treat it as string 
        exception_columns = ['MSSubClass']
        for column in self.X:
            if pd.api.types.is_numeric_dtype(self.X[column]) and column not in exception_columns:
                self.X[column] = self.X[column].fillna(X_means[column])
            else:
                self.X[column] = self.X[column].fillna(X_modes[column])
        self.y = self.y.fillna(self.y.mean())

    def str_to_categorical(self):
        """ Convert string feature into categorical feature
        Returns nothing, only updates self.X
        """
        for column in self.X:
            if not pd.api.types.is_numeric_dtype(self.X[column]):
                self.X[column] = self.X[column].astype('category')
        categorical_columns = self.X.select_dtypes(['category']).columns
        self.X[categorical_columns] = self.X[categorical_columns].apply(lambda x: x.cat.codes)
    
    def scale(self, scaler = MinMaxScaler(feature_range=(-1,1))):
        """ Scale the features, default MinMaxScaler [-1, 1]
        """
        X_data = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_data, columns=self.X.columns)
        
    def write_to_csv(self):
        self.X.to_csv(self.data_path + '_processed.csv')
