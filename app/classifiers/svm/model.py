"""
Support Vector Machine (SVM) Implementation
For Network Intrusion Detection using UNSW-NB15 Dataset
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


class SVM(BaseEstimator, ClassifierMixin):
    """
    Support Vector Machine (SVM) Classifier Wrapper
    
    SVM is a discriminative classifier that:
    - Finds the optimal hyperplane that maximizes the margin between classes
    - Uses kernel trick for non-linear decision boundaries
    - Effective in high-dimensional spaces
    - Uses sklearn's optimized SVC implementation for memory efficiency
    
    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter (penalty for misclassification)
    kernel : str, default='rbf'
        Kernel type: 'linear', 'rbf' (Gaussian), 'poly' (polynomial)
    gamma : float or str, default='scale'
        Kernel coefficient for 'rbf' and 'poly'. 
        If 'scale', uses 1 / (n_features * X.var())
        If 'auto', uses 1 / n_features
    degree : int, default=3
        Degree for polynomial kernel
    max_iter : int, default=-1
        Maximum number of iterations (-1 for no limit)
    tol : float, default=1e-3
        Tolerance for stopping criterion
    random_state : int, default=None
        Random seed for reproducibility
    cache_size : float, default=200
        Kernel cache size in MB
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3, 
                 max_iter=-1, tol=1e-3, random_state=None, cache_size=200):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cache_size = cache_size
        self._svc = None
        
    def fit(self, X, y):
        """
        Train the SVM model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Store classes for classification
        self.classes_ = np.unique(y)
        
        # Create and train sklearn's SVC (optimized for large datasets)
        self._svc = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            cache_size=self.cache_size,
            decision_function_shape='ovr'  # one-vs-rest for multi-class
        )
        
        self._svc.fit(X, y)
        
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
        
        if self._svc is None:
            raise ValueError("Model has not been fitted yet")
        
        # Get decision function from sklearn's SVC
        if len(self.classes_) == 2:
            # Binary classification - reshape to (n_samples, 2)
            scores = self._svc.decision_function(X)
            y_scores = np.column_stack([-scores, scores])
        else:
            # Multi-class - already returns (n_samples, n_classes)
            y_scores = self._svc.decision_function(X)
        
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
        if self._svc is None:
            raise ValueError("Model has not been fitted yet")
        
        return self._svc.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using decision function with softmax
        
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
        # Use softmax for probability estimation
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
        if self._svc is None:
            raise ValueError("Model has not been fitted yet")
        
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
