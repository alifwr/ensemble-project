"""
Extreme Learning Machine (ELM) Implementation
For Network Intrusion Detection using UNSW-NB15 Dataset
"""

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin


class ELM(BaseEstimator, ClassifierMixin):
    """
    Extreme Learning Machine (ELM) Classifier
    
    ELM is a single-hidden layer feedforward neural network where:
    - Input weights and biases are randomly initialized
    - Output weights are analytically calculated using Moore-Penrose pseudoinverse
    
    Parameters:
    -----------
    n_hidden : int, default=100
        Number of hidden layer neurons
    activation : str, default='sigmoid'
        Activation function: 'sigmoid', 'tanh', 'relu'
    C : float, default=1.0
        Regularization parameter (higher = less regularization)
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_hidden=100, activation='sigmoid', C=1.0, random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.C = C
        self.random_state = random_state
        
    def _activate(self, X):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return expit(X)  # Numerically stable sigmoid
        elif self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'relu':
            return np.maximum(0, X)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _compute_hidden_output(self, X):
        """Compute hidden layer output H = g(XW + b)"""
        # Linear transformation
        G = np.dot(X, self.input_weights_) + self.biases_
        # Apply activation
        H = self._activate(G)
        return H
    
    def fit(self, X, y):
        """
        Train the ELM model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            Target values
        """
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        
        # Store classes for classification
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Convert labels to one-hot encoding
        if y.ndim == 1:
            y_encoded = np.zeros((n_samples, n_classes))
            for idx, cls in enumerate(self.classes_):
                y_encoded[y == cls, idx] = 1
        else:
            y_encoded = y
        
        # Initialize random input weights and biases
        rng = np.random.RandomState(self.random_state)
        self.input_weights_ = rng.uniform(-1, 1, (n_features, self.n_hidden))
        self.biases_ = rng.uniform(-1, 1, self.n_hidden)
        
        # Compute hidden layer output matrix H
        H = self._compute_hidden_output(X)
        
        # Calculate output weights using regularized least squares
        # β = (H^T H + I/C)^(-1) H^T T
        if n_samples > self.n_hidden:
            # Use standard formula when samples > hidden neurons
            HTH = np.dot(H.T, H)
            identity = np.eye(self.n_hidden) / self.C
            self.output_weights_ = np.linalg.solve(
                HTH + identity,
                np.dot(H.T, y_encoded)
            )
        else:
            # Use alternative formula when samples < hidden neurons (more efficient)
            HHT = np.dot(H, H.T)
            identity = np.eye(n_samples) / self.C
            self.output_weights_ = np.dot(
                H.T,
                np.linalg.solve(HHT + identity, y_encoded)
            )
        
        return self
    
    def decision_function(self, X):
        """
        Compute decision function (raw predictions before argmax)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_scores : array, shape (n_samples, n_classes)
            Decision function values
        """
        X = np.asarray(X, dtype=np.float64)
        H = self._compute_hidden_output(X)
        y_scores = np.dot(H, self.output_weights_)
        return y_scores
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        y_scores = self.decision_function(X)
        y_pred_idx = np.argmax(y_scores, axis=1)
        return self.classes_[y_pred_idx]
    
    def predict_proba(self, X):
        """
        Predict class probabilities using softmax
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        proba : array, shape (n_samples, n_classes)
            Class probabilities
        """
        y_scores = self.decision_function(X)
        # Apply softmax for probabilities
        exp_scores = np.exp(y_scores - np.max(y_scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return proba
    
    def score(self, X, y, sample_weight=None):
        """
        Return accuracy score
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True labels
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights
            
        Returns:
        --------
        score : float
            Accuracy score
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)