"""
Weighted Ensemble Training Script
Network Intrusion Detection using UNSW-NB15 Dataset

This script trains an ensemble model that combines ELM, KNN, and SVM classifiers
using dynamic weighted voting based on validation accuracy.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.preprocessor import UNSWNB15Preprocessor
from app.classifiers.elm.model import ELM
from app.classifiers.knn.model import KNN
from app.classifiers.svm.model import SVM
from app.classifiers.ensembled.model import WeightedEnsemble


def main():
    """Main training function for weighted ensemble classifier"""
    
    # ========================================
    # 1. Initialize Preprocessor and Load Data
    # ========================================
    print(f'\n{"="*60}')
    print(f'INITIALIZING PREPROCESSOR')
    print(f'{"="*60}')
    
    preprocessor = UNSWNB15Preprocessor()
    
    # Define paths to data files
    data_dir = project_root / 'data'
    train_path = data_dir / 'UNSW_NB15_training-set.csv'
    test_path = data_dir / 'UNSW_NB15_testing-set.csv'
    
    # Load data
    print(f'\nLoading data...')
    X_train, X_test, y_train, y_test = preprocessor.load_data(
        train_path=str(train_path),
        test_path=str(test_path)
    )
    
    # Preprocess data
    print(f'Preprocessing data...')
    X_train_processed, y_train_encoded = preprocessor.fit_transform(X_train, y_train)
    X_test_processed, y_test_encoded = preprocessor.transform(X_test, y_test)
    
    # ========================================
    # 2. Split Training Data for Validation
    # ========================================
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_processed, y_train_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_train_encoded
    )
    
    print(f'\n{"="*60}')
    print(f'DATASET SPLIT')
    print(f'{"="*60}')
    print(f'Training samples: {X_train_split.shape[0]}')
    print(f'Validation samples: {X_val.shape[0]}')
    print(f'Testing samples: {X_test_processed.shape[0]}')
    
    # ========================================
    # 3. Dataset Information
    # ========================================
    classes = preprocessor.get_class_names()
    class_names = [str(c) for c in classes]
    features = preprocessor.get_feature_names()
    
    input_dim = X_train_processed.shape[1]
    num_classes = len(classes)
    
    print(f'\n{"="*60}')
    print(f'DATASET INFORMATION')
    print(f'{"="*60}')
    print(f'Input dimension: {input_dim}')
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {class_names}')
    
    # ========================================
    # 4. Train Base Classifiers
    # ========================================
    print(f'\n{"="*60}')
    print(f'TRAINING BASE CLASSIFIERS')
    print(f'{"="*60}')
    
    # Initialize base classifiers
    elm_model = ELM(n_hidden=1000, activation='sigmoid', C=1.0, random_state=42)
    knn_model = KNN(n_neighbors=5, weights='uniform', metric='euclidean', batch_size=1000)
    svm_model = SVM(C=1.0, kernel='rbf', gamma='scale', max_iter=-1, random_state=42, cache_size=200)
    
    base_classifiers = [
        ('ELM', elm_model),
        ('KNN', knn_model),
        ('SVM', svm_model)
    ]
    
    training_times = {}
    
    # Train each base classifier
    for name, clf in base_classifiers:
        print(f'\n{"-"*60}')
        print(f'Training {name}...')
        start_time = time.time()
        
        clf.fit(X_train_split, y_train_split)
        
        training_time = time.time() - start_time
        training_times[name] = training_time
        
        # Validation accuracy
        val_accuracy = clf.score(X_val, y_val)
        
        print(f'{name} training completed in {training_time:.2f} seconds')
        print(f'{name} validation accuracy: {val_accuracy:.4f}')
    
    print(f'\n{"-"*60}')
    print(f'All base classifiers trained successfully')
    
    # ========================================
    # 5. Create and Fit Ensemble Model
    # ========================================
    print(f'\n{"="*60}')
    print(f'CREATING WEIGHTED ENSEMBLE')
    print(f'{"="*60}')
    
    # Extract classifiers from list
    classifiers = [clf for _, clf in base_classifiers]
    
    # Create ensemble with dynamic weight calculation based on validation accuracy
    ensemble = WeightedEnsemble(
        classifiers=classifiers,
        weight_strategy='accuracy',  # Dynamic weights based on validation accuracy
        voting='soft'  # Soft voting using probabilities
    )
    
    # Fit ensemble (calculates weights based on validation performance)
    print(f'\nCalculating dynamic weights based on validation accuracy...')
    ensemble.fit(X_train_split, y_train_split, validation_data=(X_val, y_val))
    
    # Display calculated weights
    print(f'\n{"-"*60}')
    print(f'CALCULATED WEIGHTS')
    print(f'{"-"*60}')
    weights = ensemble.get_weights()
    classifier_names = ['ELM', 'KNN', 'SVM']
    for i, (clf_key, weight) in enumerate(weights.items()):
        print(f'{classifier_names[i]:<10}: {weight:.4f}')
    print(f'{"-"*60}')
    
    # ========================================
    # 6. Make Predictions
    # ========================================
    print(f'\n{"="*60}')
    print(f'GENERATING PREDICTIONS')
    print(f'{"="*60}')
    
    # Predictions on validation set
    print(f'\nPredicting validation set...')
    y_val_pred = ensemble.predict(X_val)
    
    # Predictions on test set
    if y_test_encoded is not None:
        print(f'Predicting test set...')
        y_test_pred = ensemble.predict(X_test_processed)
    else:
        raise ValueError("y_test_encoded is None - labels are required for evaluation")
    
    print(f'Predictions completed')
    
    # ========================================
    # 7. Calculate Performance Metrics
    # ========================================
    # Calculate metrics for validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    
    # Calculate metrics for test set
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    test_precision = precision_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
    
    # ========================================
    # 8. Display Overall Performance Metrics
    # ========================================
    print(f'\n{"="*60}')
    print(f'MODEL EVALUATION')
    print(f'{"="*60}')
    print(f'\nOverall Performance Metrics:')
    print(f'{"-"*60}')
    print(f'{"Metric":<20} {"Validation":<20} {"Testing":<20}')
    print(f'{"-"*60}')
    print(f'{"Accuracy":<20} {val_accuracy:<20.4f} {test_accuracy:<20.4f}')
    print(f'{"Precision":<20} {val_precision:<20.4f} {test_precision:<20.4f}')
    print(f'{"Recall":<20} {val_recall:<20.4f} {test_recall:<20.4f}')
    print(f'{"F1-Score":<20} {val_f1:<20.4f} {test_f1:<20.4f}')
    print(f'{"-"*60}')
    
    # ========================================
    # 9. Compare with Base Classifiers
    # ========================================
    print(f'\n{"="*60}')
    print(f'COMPARISON WITH BASE CLASSIFIERS (Test Set)')
    print(f'{"="*60}')
    print(f'\n{"Classifier":<20} {"Accuracy":<15} {"Precision":<15} {"Recall":<15} {"F1-Score":<15}')
    print(f'{"-"*85}')
    
    # Evaluate base classifiers on test set
    for name, clf in base_classifiers:
        y_pred = clf.predict(X_test_processed)
        acc = accuracy_score(y_test_encoded, y_pred)
        prec = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        
        print(f'{name:<20} {acc:<15.4f} {prec:<15.4f} {rec:<15.4f} {f1:<15.4f}')
    
    # Display ensemble performance
    print(f'{"-"*85}')
    print(f'{"Ensemble":<20} {test_accuracy:<15.4f} {test_precision:<15.4f} {test_recall:<15.4f} {test_f1:<15.4f}')
    print(f'{"-"*85}')
    
    # ========================================
    # 10. Detailed Classification Report
    # ========================================
    print(f'\n{"="*60}')
    print(f'DETAILED CLASSIFICATION REPORT (Test Set)')
    print(f'{"="*60}')
    print(classification_report(y_test_encoded, y_test_pred, target_names=class_names))
    
    # ========================================
    # 11. Confusion Matrix
    # ========================================
    print(f'\n{"="*60}')
    print(f'CONFUSION MATRIX (Test Set)')
    print(f'{"="*60}')
    cm = confusion_matrix(y_test_encoded, y_test_pred)
    print(f'\nRows: True labels, Columns: Predicted labels')
    print(f'Classes: {class_names}\n')
    print(cm)
    
    # ========================================
    # 12. Per-Class Accuracy
    # ========================================
    print(f'\n{"="*60}')
    print(f'PER-CLASS ACCURACY (Test Set)')
    print(f'{"="*60}')
    for i, class_name in enumerate(class_names):
        class_mask = (y_test_encoded == i)
        num_samples = int(np.sum(class_mask))
        if num_samples > 0:
            class_accuracy = float(np.sum(y_test_pred[class_mask] == i)) / num_samples
            print(f'{class_name:<20}: {class_accuracy:.4f} ({num_samples} samples)')
    
    # ========================================
    # 13. Save Report to File
    # ========================================
    print(f'\n{"="*60}')
    print(f'SAVING REPORT')
    print(f'{"="*60}')
    
    report_dir = project_root / 'results'
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / 'ensemble_classification_report.txt'
    
    with open(report_file, 'w') as f:
        f.write(f'{"="*60}\n')
        f.write(f'WEIGHTED ENSEMBLE CLASSIFICATION REPORT\n')
        f.write(f'{"="*60}\n\n')
        
        f.write(f'Training Date: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write(f'{"="*60}\n')
        f.write(f'ENSEMBLE CONFIGURATION\n')
        f.write(f'{"="*60}\n')
        f.write(f'Base Classifiers: ELM, KNN, SVM\n')
        f.write(f'Weight Strategy: {ensemble.weight_strategy}\n')
        f.write(f'Voting: {ensemble.voting}\n\n')
        
        f.write(f'{"="*60}\n')
        f.write(f'CLASSIFIER WEIGHTS\n')
        f.write(f'{"="*60}\n')
        for i, (clf_key, weight) in enumerate(weights.items()):
            f.write(f'{classifier_names[i]:<10}: {weight:.4f}\n')
        f.write('\n')
        
        f.write(f'{"="*60}\n')
        f.write(f'BASE CLASSIFIER TRAINING TIMES\n')
        f.write(f'{"="*60}\n')
        for name, train_time in training_times.items():
            f.write(f'{name:<10}: {train_time:.2f} seconds\n')
        f.write('\n')
        
        f.write(f'{"="*60}\n')
        f.write(f'DATASET INFORMATION\n')
        f.write(f'{"="*60}\n')
        f.write(f'Input dimension: {input_dim}\n')
        f.write(f'Number of classes: {num_classes}\n')
        f.write(f'Training samples: {X_train_split.shape[0]}\n')
        f.write(f'Validation samples: {X_val.shape[0]}\n')
        f.write(f'Testing samples: {X_test_processed.shape[0]}\n')
        f.write(f'Classes: {", ".join(class_names)}\n\n')
        
        f.write(f'{"="*60}\n')
        f.write(f'OVERALL PERFORMANCE METRICS\n')
        f.write(f'{"="*60}\n')
        f.write(f'{"Metric":<20} {"Validation":<20} {"Testing":<20}\n')
        f.write(f'{"-"*60}\n')
        f.write(f'{"Accuracy":<20} {val_accuracy:<20.4f} {test_accuracy:<20.4f}\n')
        f.write(f'{"Precision":<20} {val_precision:<20.4f} {test_precision:<20.4f}\n')
        f.write(f'{"Recall":<20} {val_recall:<20.4f} {test_recall:<20.4f}\n')
        f.write(f'{"F1-Score":<20} {val_f1:<20.4f} {test_f1:<20.4f}\n')
        f.write(f'{"-"*60}\n\n')
        
        f.write(f'{"="*60}\n')
        f.write(f'COMPARISON WITH BASE CLASSIFIERS (Test Set)\n')
        f.write(f'{"="*60}\n')
        f.write(f'{"Classifier":<20} {"Accuracy":<15} {"Precision":<15} {"Recall":<15} {"F1-Score":<15}\n')
        f.write(f'{"-"*85}\n')
        
        for name, clf in base_classifiers:
            y_pred = clf.predict(X_test_processed)
            acc = accuracy_score(y_test_encoded, y_pred)
            prec = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            f.write(f'{name:<20} {acc:<15.4f} {prec:<15.4f} {rec:<15.4f} {f1:<15.4f}\n')
        
        f.write(f'{"-"*85}\n')
        f.write(f'{"Ensemble":<20} {test_accuracy:<15.4f} {test_precision:<15.4f} {test_recall:<15.4f} {test_f1:<15.4f}\n')
        f.write(f'{"-"*85}\n\n')
        
        f.write(f'{"="*60}\n')
        f.write(f'DETAILED CLASSIFICATION REPORT (Test Set)\n')
        f.write(f'{"="*60}\n')
        report_str = classification_report(y_test_encoded, y_test_pred, target_names=class_names)
        f.write(str(report_str))
        f.write('\n')
        
        f.write(f'{"="*60}\n')
        f.write(f'CONFUSION MATRIX (Test Set)\n')
        f.write(f'{"="*60}\n')
        f.write(f'Rows: True labels, Columns: Predicted labels\n')
        f.write(f'Classes: {", ".join(class_names)}\n\n')
        f.write(str(cm))
        f.write('\n\n')
        
        f.write(f'{"="*60}\n')
        f.write(f'PER-CLASS ACCURACY (Test Set)\n')
        f.write(f'{"="*60}\n')
        for i, class_name in enumerate(class_names):
            class_mask = (y_test_encoded == i)
            num_samples = int(np.sum(class_mask))
            if num_samples > 0:
                class_accuracy = float(np.sum(y_test_pred[class_mask] == i)) / num_samples
                f.write(f'{class_name:<20}: {class_accuracy:.4f} ({num_samples} samples)\n')
    
    print(f'\nReport saved to: {report_file}')
    print(f'\n{"="*60}')
    print(f'ENSEMBLE TRAINING COMPLETE')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
