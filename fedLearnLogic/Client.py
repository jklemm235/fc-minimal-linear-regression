from typing import Tuple, List
import yaml
import pandas as pd
from numpy import ndarray

class Client:
    """Client class to handle local data and computations for a federated linear regression"""
    def __init__(self, inputfolder: str):
        self.inputfolder = inputfolder
        config = self._read_config()
        self.datafile = config["datafile"]
        self.separator = config["separator"]
        self.target = config["target"]

        # read in the data
        self.data = pd.read_csv(f"{self.inputfolder}/{self.datafile}", sep=self.separator)
        self.X = self.data.drop(columns=[self.target])
        self.X["intercept"] = 1.0  # add intercept term
        self.y = self.data[self.target]

    def get_feature_names(self):
        """Return the variable names of the features"""
        features = self.X.columns.tolist()
        return features

    def update_to_common_features(self, common_features: List[str]):
        """Update the local data to only include the common features."""
        self.X = self.X[common_features]
        # sort to ensure all clients have the same order of features
        self.X = self.X.sort_index(axis=1)

    def calculate_XtX(self):
        """Calculate X^T * X."""
        return self.X.T @ self.X

    def calculate_Xty(self):
        """Calculate X^T * y."""
        return self.X.T @ self.y

    def save_model(self, global_beta: ndarray, outputfolder: str):
        """Save the model parameters to a yaml file.
        The yaml file keys are the feature names and the values are the corresponding beta coefficients.
        Args:
            global_beta: The global beta coefficients calculated by the coordinator.
            outputfolder: The folder where the model parameters should be saved.
        Returns:
            None
        """
        model = {}
        for feature, beta in zip(self.X.columns, global_beta):
            model[feature] = beta
        with open(f"{outputfolder}/model_params.yaml", "w") as f:
            yaml.dump(model, f)


    def _read_config(self) -> dict:
        """ Read config.yaml, validate it and return the config as a dictionary.
        Args:
            inputfolder: The folder where the config.yaml is located.
        Returns:
            A dictionary containing the config parameters:
            - datafile: The name of the data file (e.g., "data.csv").
            - separator: The separator used in the data file (e.g., ",").
            - target: The name of the target variable in the data file (e.g., "target").
        Raises:
            ValueError: If the config file is missing required sections or parameters.
        """
        with open(f"{self.inputfolder}/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        # small validation
        if "LinearRegressionApp" not in config:
            raise ValueError("Config file must contain 'LinearRegressionApp' section.")
        if "datafile" not in config["LinearRegressionApp"]:
            raise ValueError("Config file must contain 'datafile' in 'LinearRegressionApp' section.")
        if "separator" not in config["LinearRegressionApp"]:
            raise ValueError("Config file must contain 'separator' in 'LinearRegressionApp' section.")
        if "target" not in config["LinearRegressionApp"]:
            raise ValueError("Config file must contain 'target' in 'LinearRegressionApp' section.")

        return config["LinearRegressionApp"]