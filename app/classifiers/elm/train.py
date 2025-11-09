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
import time

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from app.utils.preprocessor import UNSWNB15Preprocessor
from app.classifiers.elm.model import ELM

preprocessor = UNSWNB15Preprocessor()
model = ELM(n_hidden=100, activation='sigmoid')

# Define paths to data files
data_dir = project_root / 'data'
train_path = data_dir / 'UNSW_NB15_training-set.csv'
test_path = data_dir / 'UNSW_NB15_testing-set.csv'

X_train, X_test, y_train, y_test = preprocessor.load_data(
    train_path=str(train_path),
    test_path=str(test_path)
)

X_train_processed, y_train_encoded = preprocessor.fit_transform(X_train, y_train)
X_test_processed, y_test_encoded = preprocessor.transform(X_test, y_test)

classes = preprocessor.get_class_names()
# Convert class names to strings for reporting
class_names = [str(c) for c in classes]
features = preprocessor.get_feature_names()

input_dim = X_train_processed.shape[1]
num_classes = len(classes)
print(f'\n{"="*60}')
print(f'DATASET INFORMATION')
print(f'{"="*60}')
print(f'Input dimension: {input_dim}')
print(f'Number of classes: {num_classes}')
print(f'Training samples: {X_train_processed.shape[0]}')
print(f'Testing samples: {X_test_processed.shape[0]}')
print(f'Classes: {class_names}')

# Train
print(f'\n{"="*60}')
print(f'TRAINING ELM MODEL')
print(f'{"="*60}')
print(f'Hidden neurons: {model.n_hidden}')
print(f'Activation function: {model.activation}')
print(f'Regularization (C): {model.C}')
print(f'\nTraining started...')
start_time = time.time()
model.fit(X_train_processed, y_train_encoded)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Evaluate
print(f'\n{"="*60}')
print(f'MODEL EVALUATION')
print(f'{"="*60}')

# Make predictions
y_train_pred = model.predict(X_train_processed)
if y_test_encoded is not None:
    y_test_pred = model.predict(X_test_processed)
else:
    raise ValueError("y_test_encoded is None - labels are required for evaluation")

# Calculate metrics
train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
test_accuracy = accuracy_score(y_test_encoded, y_test_pred)

train_precision = precision_score(y_train_encoded, y_train_pred, average='weighted', zero_division=0)
test_precision = precision_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)

train_recall = recall_score(y_train_encoded, y_train_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)

train_f1 = f1_score(y_train_encoded, y_train_pred, average='weighted', zero_division=0)
test_f1 = f1_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)

# Print summary metrics
print(f'\nOverall Performance Metrics:')
print(f'{"-"*60}')
print(f'{"Metric":<20} {"Training":<20} {"Testing":<20}')
print(f'{"-"*60}')
print(f'{"Accuracy":<20} {train_accuracy:<20.4f} {test_accuracy:<20.4f}')
print(f'{"Precision":<20} {train_precision:<20.4f} {test_precision:<20.4f}')
print(f'{"Recall":<20} {train_recall:<20.4f} {test_recall:<20.4f}')
print(f'{"F1-Score":<20} {train_f1:<20.4f} {test_f1:<20.4f}')
print(f'{"-"*60}')

# Detailed classification report
print(f'\n{"="*60}')
print(f'DETAILED CLASSIFICATION REPORT (Test Set)')
print(f'{"="*60}')
print(classification_report(y_test_encoded, y_test_pred, target_names=class_names))

# Confusion Matrix
print(f'\n{"="*60}')
print(f'CONFUSION MATRIX (Test Set)')
print(f'{"="*60}')
cm = confusion_matrix(y_test_encoded, y_test_pred)
print(f'\nRows: True labels, Columns: Predicted labels')
print(f'Classes: {class_names}\n')
print(cm)

# Per-class accuracy
print(f'\n{"="*60}')
print(f'PER-CLASS ACCURACY (Test Set)')
print(f'{"="*60}')
for i, class_name in enumerate(class_names):
    class_mask = (y_test_encoded == i)
    num_samples = int(np.sum(class_mask))  # type: ignore
    if num_samples > 0:
        class_accuracy = float(np.sum(y_test_pred[class_mask] == i)) / num_samples  # type: ignore
        print(f'{class_name:<20}: {class_accuracy:.4f} ({num_samples} samples)')

# Save report to file
print(f'\n{"="*60}')
print(f'SAVING REPORT')
print(f'{"="*60}')
report_dir = project_root / 'results'
report_dir.mkdir(exist_ok=True)
report_file = report_dir / 'elm_classification_report.txt'

with open(report_file, 'w') as f:
    f.write(f'{"="*60}\n')
    f.write(f'ELM CLASSIFICATION REPORT\n')
    f.write(f'{"="*60}\n\n')
    
    f.write(f'Training Date: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'Training Time: {training_time:.2f} seconds\n\n')
    
    f.write(f'{"="*60}\n')
    f.write(f'MODEL CONFIGURATION\n')
    f.write(f'{"="*60}\n')
    f.write(f'Hidden neurons: {model.n_hidden}\n')
    f.write(f'Activation function: {model.activation}\n')
    f.write(f'Regularization (C): {model.C}\n\n')
    
    f.write(f'{"="*60}\n')
    f.write(f'DATASET INFORMATION\n')
    f.write(f'{"="*60}\n')
    f.write(f'Input dimension: {input_dim}\n')
    f.write(f'Number of classes: {num_classes}\n')
    f.write(f'Training samples: {X_train_processed.shape[0]}\n')
    f.write(f'Testing samples: {X_test_processed.shape[0]}\n')
    f.write(f'Classes: {", ".join(class_names)}\n\n')
    
    f.write(f'{"="*60}\n')
    f.write(f'OVERALL PERFORMANCE METRICS\n')
    f.write(f'{"="*60}\n')
    f.write(f'{"Metric":<20} {"Training":<20} {"Testing":<20}\n')
    f.write(f'{"-"*60}\n')
    f.write(f'{"Accuracy":<20} {train_accuracy:<20.4f} {test_accuracy:<20.4f}\n')
    f.write(f'{"Precision":<20} {train_precision:<20.4f} {test_precision:<20.4f}\n')
    f.write(f'{"Recall":<20} {train_recall:<20.4f} {test_recall:<20.4f}\n')
    f.write(f'{"F1-Score":<20} {train_f1:<20.4f} {test_f1:<20.4f}\n')
    f.write(f'{"-"*60}\n\n')
    
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
        num_samples = int(np.sum(class_mask))  # type: ignore
        if num_samples > 0:
            class_accuracy = float(np.sum(y_test_pred[class_mask] == i)) / num_samples  # type: ignore
            f.write(f'{class_name:<20}: {class_accuracy:.4f} ({num_samples} samples)\n')

print(f'\nReport saved to: {report_file}')
print(f'\n{"="*60}')
print(f'TRAINING COMPLETE')
print(f'{"="*60}')
