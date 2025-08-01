import numpy as np
from collections import Counter

class TestModel:
    def __init__(self, behavior='constant', constant_value=0):
        """
        Initialize the TestModel with a specific behavior.
        
        Parameters:
        - behavior (str): The behavior of the model ('constant', 'mean', 'mode', 'identity').
        - constant_value (float or int): The value to predict if behavior is 'constant'.
        """
        self.behavior = behavior
        self.constant_value = constant_value
        self.input_history = []
        self.output_history = []
        self.fitted_value = None

    def fit(self, X, y):
        """
        Fit the model based on the specified behavior.
        
        Parameters:
        - X: Input features (not used for 'constant', 'mean', 'mode').
        - y: Target values (used for 'mean' and 'mode').
        """
        if self.behavior == 'mean':
            self.fitted_value = np.mean(y)
        elif self.behavior == 'mode':
            self.fitted_value = Counter(y).most_common(1)[0][0]
        elif self.behavior == 'constant':
            self.fitted_value = self.constant_value
        else:
            self.fitted_value = None

    def predict(self, X):
        """
        Make predictions based on the specified behavior.
        
        Parameters:
        - X: Input features.
        
        Returns:
        - Predictions based on the model's behavior.
        """
        self.input_history.append(X)
        
        if self.behavior in ['constant', 'mean', 'mode']:
            if self.fitted_value is not None:
                output = [self.fitted_value] * len(X)
            else:
                raise ValueError("Model not fitted yet")
        elif self.behavior == 'identity':
            output = X
        else:
            raise ValueError("Unsupported behavior")
        
        self.output_history.append(output)
        return output