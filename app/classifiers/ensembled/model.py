"""
Weighted Ensemble Classifier Implementation
Combines ELM, KNN, and SVM using dynamic weighted voting
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Optional, Dict
import warnings


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted Ensemble Classifier with Dynamic Weight Adjustment
    
    Combines predictions from multiple base classifiers (ELM, KNN, SVM) using
    weighted voting. Weights can be set manually or calculated dynamically based
    on individual classifier performance on validation data.
    
    Parameters:
    -----------
    classifiers : list
        List of fitted base classifiers (must implement predict and predict_proba)
    weights : array-like, optional, default=None
        Manual weights for each classifier. If None, uses equal weights initially.
        Weights will be normalized to sum to 1.0
    weight_strategy : str, default='accuracy'
        Strategy for dynamic weight calculation:
        - 'equal': Equal weights for all classifiers
        - 'accuracy': Weights based on validation accuracy
        - 'f1': Weights based on validation F1-score
        - 'manual': Use manually specified weights (no dynamic adjustment)
    voting : str, default='soft'
        Voting strategy:
        - 'soft': Use predicted probabilities (weighted average)
        - 'hard': Use predicted class labels (weighted majority vote)
    
    Attributes:
    -----------
    classifiers_ : list
        Fitted base classifiers
    weights_ : ndarray
        Normalized weights for each classifier
    classes_ : ndarray
        Unique class labels
    n_classifiers_ : int
        Number of base classifiers
    """
    
    def __init__(self, classifiers=None, weights=None, weight_strategy='accuracy', voting='soft'):
        self.classifiers = classifiers if classifiers is not None else []
        self.weights = weights
        self.weight_strategy = weight_strategy
        self.voting = voting
        
    def fit(self, X, y, validation_data=None):
        """
        Fit the ensemble by calculating optimal weights
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data (not used if classifiers are already fitted)
        y : array-like, shape (n_samples,)
            Target labels (not used if classifiers are already fitted)
        validation_data : tuple (X_val, y_val), optional
            Validation data for dynamic weight calculation
            If None and weight_strategy != 'manual', uses training data
        
        Returns:
        --------
        self : WeightedEnsemble
            Fitted ensemble
        """
        if len(self.classifiers) == 0:
            raise ValueError("No classifiers provided. Add classifiers before fitting.")
        
        self.classifiers_ = self.classifiers
        self.n_classifiers_ = len(self.classifiers_)
        
        # Get classes from first classifier
        if hasattr(self.classifiers_[0], 'classes_'):
            self.classes_ = self.classifiers_[0].classes_
        else:
            self.classes_ = np.unique(y)
        
        # Calculate weights based on strategy
        if self.weight_strategy == 'manual':
            if self.weights is None:
                warnings.warn("Manual weight strategy but no weights provided. Using equal weights.")
                self.weights_ = np.ones(self.n_classifiers_) / self.n_classifiers_
            else:
                self.weights_ = np.array(self.weights)
                self.weights_ = self.weights_ / np.sum(self.weights_)  # Normalize
        elif self.weight_strategy == 'equal':
            self.weights_ = np.ones(self.n_classifiers_) / self.n_classifiers_
        else:
            # Dynamic weight calculation
            if validation_data is None:
                warnings.warn("No validation data provided. Using training data for weight calculation.")
                X_val, y_val = X, y
            else:
                X_val, y_val = validation_data
            
            self.weights_ = self._calculate_dynamic_weights(X_val, y_val)
        
        return self
    
    def _calculate_dynamic_weights(self, X_val, y_val):
        """
        Calculate dynamic weights based on individual classifier performance
        
        Parameters:
        -----------
        X_val : array-like, shape (n_samples, n_features)
            Validation features
        y_val : array-like, shape (n_samples,)
            Validation labels
        
        Returns:
        --------
        weights : ndarray
            Normalized weights for each classifier
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        performance_scores = []
        
        for clf in self.classifiers_:
            try:
                y_pred = clf.predict(X_val)
                
                if self.weight_strategy == 'accuracy':
                    score = accuracy_score(y_val, y_pred)
                elif self.weight_strategy == 'f1':
                    score = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                else:
                    score = accuracy_score(y_val, y_pred)  # Default to accuracy
                
                performance_scores.append(score)
            except Exception as e:
                warnings.warn(f"Error calculating performance for classifier: {e}. Using score 0.")
                performance_scores.append(0.0)
        
        performance_scores = np.array(performance_scores)
        
        # Avoid division by zero
        if np.sum(performance_scores) == 0:
            warnings.warn("All classifiers have zero performance. Using equal weights.")
            return np.ones(self.n_classifiers_) / self.n_classifiers_
        
        # Normalize to get weights
        weights = performance_scores / np.sum(performance_scores)
        
        return weights
    
    def predict_proba(self, X):
        """
        Predict class probabilities using weighted voting
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        
        Returns:
        --------
        proba : ndarray, shape (n_samples, n_classes)
            Weighted average of predicted probabilities
        """
        if not hasattr(self, 'classifiers_'):
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize probability matrix
        weighted_proba = np.zeros((n_samples, n_classes))
        
        for idx, clf in enumerate(self.classifiers_):
            try:
                # Get probabilities from classifier
                if hasattr(clf, 'predict_proba'):
                    proba = clf.predict_proba(X)
                else:
                    # If classifier doesn't have predict_proba, use one-hot encoding of predictions
                    predictions = clf.predict(X)
                    proba = np.zeros((n_samples, n_classes))
                    for i, pred in enumerate(predictions):
                        proba[i, int(pred)] = 1.0
                
                # Add weighted probabilities
                weighted_proba += self.weights_[idx] * proba
                
            except Exception as e:
                warnings.warn(f"Error getting predictions from classifier {idx}: {e}. Skipping.")
                continue
        
        return weighted_proba
    
    def predict(self, X):
        """
        Predict class labels using weighted voting
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        
        Returns:
        --------
        predictions : ndarray, shape (n_samples,)
            Predicted class labels
        """
        if self.voting == 'soft':
            # Use probability-based voting
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:
            # Hard voting - weighted majority vote
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            
            # Collect votes
            votes = np.zeros((n_samples, n_classes))
            
            for idx, clf in enumerate(self.classifiers_):
                try:
                    predictions = clf.predict(X)
                    for i, pred in enumerate(predictions):
                        votes[i, int(pred)] += self.weights_[idx]
                except Exception as e:
                    warnings.warn(f"Error getting predictions from classifier {idx}: {e}. Skipping.")
                    continue
            
            # Return class with highest weighted vote
            return np.argmax(votes, axis=1)
    
    def decision_function(self, X):
        """
        Compute decision function for each sample
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        
        Returns:
        --------
        decision : ndarray, shape (n_samples, n_classes)
            Decision function values (same as predict_proba for ensemble)
        """
        return self.predict_proba(X)
    
    def score(self, X, y):
        """
        Calculate accuracy score on test data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels
        
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        from sklearn.metrics import accuracy_score
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_weights(self):
        """
        Get current classifier weights
        
        Returns:
        --------
        weights : dict
            Dictionary mapping classifier index to weight
        """
        if not hasattr(self, 'weights_'):
            return None
        
        return {
            f'Classifier_{i}': weight 
            for i, weight in enumerate(self.weights_)
        }
    
    def get_classifier_names(self):
        """
        Get names of base classifiers
        
        Returns:
        --------
        names : list
            List of classifier class names
        """
        return [clf.__class__.__name__ for clf in self.classifiers_]
