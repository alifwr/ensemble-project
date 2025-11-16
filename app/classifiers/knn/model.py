"""
K-Nearest Neighbors (KNN) Implementation
For Network Intrusion Detection using UNSW-NB15 Dataset
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import mode


class KNN(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors (KNN) Classifier
    
    KNN is a non-parametric classification algorithm that:
    - Stores all training examples
    - Predicts by finding k nearest neighbors and majority voting
    - Uses batch processing to handle large datasets efficiently
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use for classification
    weights : str, default='uniform'
        Weight function: 'uniform' (equal weights) or 'distance' (inverse distance)
    metric : str, default='euclidean'
        Distance metric: 'euclidean', 'manhattan', 'minkowski'
    p : int, default=2
        Power parameter for Minkowski metric (p=1: manhattan, p=2: euclidean)
    batch_size : int, default=1000
        Number of samples to process at once (prevents memory overflow)
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', p=2, batch_size=1000):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.batch_size = batch_size
        
    def _compute_distances_batch(self, X_test):
        """Compute distances between test samples and training samples in batches"""
        if self.metric == 'euclidean':
            return euclidean_distances(X_test, self.X_train_)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X_test[:, np.newaxis] - self.X_train_), axis=2)
        elif self.metric == 'minkowski':
            return np.sum(np.abs(X_test[:, np.newaxis] - self.X_train_) ** self.p, axis=2) ** (1/self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X, y):
        """
        Store training data
        
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
        
        # Store training data
        self.X_train_ = X
        self.y_train_ = y
        
        # Store classes for classification
        self.classes_ = np.unique(y)
        
        return self
    
    def _get_neighbors(self, distances):
        """Get k nearest neighbors indices and distances"""
        # Get indices of k smallest distances
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        return neighbor_indices, neighbor_distances
    
    def decision_function(self, X):
        """
        Compute decision function (raw predictions before argmax)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        ---------
        y_scores : array, shape (n_samples, n_classes)
            Decision function values (weighted votes)
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        y_scores = np.zeros((n_samples, n_classes))
        
        # Process in batches to avoid memory issues
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            
            # Compute distances to all training samples for this batch
            distances = self._compute_distances_batch(X_batch)
            
            # Get k nearest neighbors
            neighbor_indices, neighbor_distances = self._get_neighbors(distances)
            
            # Get neighbor labels
            neighbor_labels = self.y_train_[neighbor_indices]
            
            # Compute weighted votes for this batch
            batch_size_actual = end_idx - start_idx
            for i in range(batch_size_actual):
                if self.weights == 'uniform':
                    # Count occurrences of each class
                    for j, cls in enumerate(self.classes_):
                        y_scores[start_idx + i, j] = np.sum(neighbor_labels[i] == cls)
                elif self.weights == 'distance':
                    # Weight by inverse distance
                    for j, cls in enumerate(self.classes_):
                        mask = neighbor_labels[i] == cls
                        if np.any(mask):
                            weights = 1.0 / (neighbor_distances[i, mask] + 1e-10)  # Add small epsilon
                            y_scores[start_idx + i, j] = np.sum(weights)
                else:
                    raise ValueError(f"Unknown weights: {self.weights}")
        
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
        Predict class probabilities
        
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
        # Normalize scores to probabilities
        proba = y_scores / np.sum(y_scores, axis=1, keepdims=True)
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
