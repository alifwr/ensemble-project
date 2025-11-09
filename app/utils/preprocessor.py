"""
Data Preprocessing Pipeline for UNSW-NB15 Dataset
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class UNSWNB15Preprocessor:
    """
    Preprocessing pipeline for UNSW-NB15 Network Intrusion Detection Dataset
    
    The UNSW-NB15 dataset contains network traffic features with both normal
    and attack samples across multiple attack categories.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.categorical_encoders = {}
        self.feature_names = None
        self.is_fitted = False
        
        # Define UNSW-NB15 feature columns
        self.feature_columns = [
            'dur', 'proto', 'service', 'state', 'spkts', 'dpkts',
            'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
            'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
            'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
            'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
            'response_body_len', 'ct_srv_src', 'ct_state_ttl',
            'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
            'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
            'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
            'is_sm_ips_ports'
        ]
        
        # Categorical features
        self.categorical_features = ['proto', 'service', 'state']
        
        # Numerical features
        self.numerical_features = [f for f in self.feature_columns 
                                   if f not in self.categorical_features]
    
    def load_data(self, train_path=None, test_path=None, csv_path=None):
        """
        Load UNSW-NB15 dataset
        
        Parameters:
        -----------
        train_path : str, optional
            Path to training CSV file
        test_path : str, optional
            Path to testing CSV file
        csv_path : str, optional
            Path to single CSV file (will be split into train/test)
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Preprocessed training and testing data
        """
        if csv_path:
            print(f"Loading data from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Split into features and labels
            if 'label' in df.columns:
                y = df['label']
                X = df.drop(['label'], axis=1, errors='ignore')
            elif 'attack_cat' in df.columns:
                # Use attack category as label
                y = df['attack_cat']
                X = df.drop(['attack_cat', 'label'], axis=1, errors='ignore')
            else:
                raise ValueError("No label column found in dataset")
            
            # Remove any ID columns
            X = X.drop(['id'], axis=1, errors='ignore')
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
        elif train_path and test_path:
            print(f"Loading training data from: {train_path}")
            print(f"Loading testing data from: {test_path}")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Extract labels
            y_train = train_df['label'] if 'label' in train_df.columns else train_df['attack_cat']
            y_test = test_df['label'] if 'label' in test_df.columns else test_df['attack_cat']
            
            # Extract features
            X_train = train_df.drop(['label', 'attack_cat', 'id'], axis=1, errors='ignore')
            X_test = test_df.drop(['label', 'attack_cat', 'id'], axis=1, errors='ignore')
        else:
            raise ValueError("Must provide either csv_path or both train_path and test_path")
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {y_train.nunique()}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_missing_values(self, X):
        """Handle missing values in the dataset"""
        # Replace inf values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill numerical features with median
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        # Fill categorical features with mode
        for col in X.select_dtypes(include=['object']).columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].mode()[0], inplace=True)
        
        return X
    
    def encode_categorical(self, X, fit=True):
        """Encode categorical features"""
        X = X.copy()
        
        for col in X.select_dtypes(include=['object']).columns:
            if fit:
                # Fit and transform
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.categorical_encoders[col] = le
            else:
                # Transform only
                le = self.categorical_encoders[col]
                # Handle unseen categories
                def encode_value(x):
                    if x in le.classes_:
                        result = le.transform([x])
                        return int(result[0])  # type: ignore
                    return -1
                X[col] = X[col].astype(str).map(encode_value)
        
        return X
    
    def remove_outliers(self, X, y=None, contamination=0.1):
        """
        Remove outliers using IQR method (optional)
        
        Parameters:
        -----------
        X : DataFrame
            Feature data
        y : Series, optional
            Labels (will be filtered accordingly)
        contamination : float
            Proportion of outliers to remove
        """
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create mask for non-outliers
        mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
        
        X_clean = X[mask]
        y_clean = y[mask] if y is not None else None
        
        print(f"Removed {(~mask).sum()} outliers ({(~mask).sum() / len(X) * 100:.2f}%)")
        
        return X_clean, y_clean
    
    def create_features(self, X):
        """
        Create additional engineered features
        
        Engineering features specific to network traffic analysis
        """
        X = X.copy()
        
        # Packet size ratios
        if 'sbytes' in X.columns and 'dbytes' in X.columns:
            X['byte_ratio'] = X['sbytes'] / (X['dbytes'] + 1)
            X['total_bytes'] = X['sbytes'] + X['dbytes']
        
        # Packet count ratios
        if 'spkts' in X.columns and 'dpkts' in X.columns:
            X['pkt_ratio'] = X['spkts'] / (X['dpkts'] + 1)
            X['total_pkts'] = X['spkts'] + X['dpkts']
        
        # Load features
        if 'sload' in X.columns and 'dload' in X.columns:
            X['load_ratio'] = X['sload'] / (X['dload'] + 1)
            X['total_load'] = X['sload'] + X['dload']
        
        # Loss features
        if 'sloss' in X.columns and 'dloss' in X.columns:
            X['total_loss'] = X['sloss'] + X['dloss']
        
        # TTL features
        if 'sttl' in X.columns and 'dttl' in X.columns:
            X['ttl_diff'] = abs(X['sttl'] - X['dttl'])
        
        # Inter-packet features
        if 'sinpkt' in X.columns and 'dinpkt' in X.columns:
            X['inpkt_ratio'] = X['sinpkt'] / (X['dinpkt'] + 1)
        
        # Jitter features
        if 'sjit' in X.columns and 'djit' in X.columns:
            X['jit_diff'] = abs(X['sjit'] - X['djit'])
        
        return X
    
    def fit_transform(self, X_train, y_train):
        """
        Fit the preprocessor and transform training data
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training labels
            
        Returns:
        --------
        X_processed : array
            Preprocessed features
        y_processed : array
            Encoded labels
        """
        print("\n=== Preprocessing Training Data ===")
        
        # Handle missing values
        print("Handling missing values...")
        X_train = self.handle_missing_values(X_train)
        
        # Encode categorical features
        print("Encoding categorical features...")
        X_train = self.encode_categorical(X_train, fit=True)
        
        # Create engineered features
        print("Creating engineered features...")
        X_train = self.create_features(X_train)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Scale numerical features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode labels
        print("Encoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        self.is_fitted = True
        print(f"Final feature dimension: {X_train_scaled.shape[1]}")
        print("Preprocessing complete!")
        
        return X_train_scaled, y_train_encoded
    
    def transform(self, X_test, y_test=None):
        """
        Transform test data using fitted preprocessor
        
        Parameters:
        -----------
        X_test : DataFrame
            Test features
        y_test : Series, optional
            Test labels
            
        Returns:
        --------
        X_processed : array
            Preprocessed features
        y_processed : array or None
            Encoded labels (if y_test provided)
        """
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        print("\n=== Preprocessing Test Data ===")
        
        # Handle missing values
        X_test = self.handle_missing_values(X_test)
        
        # Encode categorical features
        X_test = self.encode_categorical(X_test, fit=False)
        
        # Create engineered features
        X_test = self.create_features(X_test)
        
        # Ensure same features as training
        for col in self.feature_names:
            if col not in X_test.columns:
                X_test[col] = 0
        
        X_test = X_test[self.feature_names]
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels if provided
        y_test_encoded = None
        if y_test is not None:
            y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"Test samples processed: {X_test_scaled.shape[0]}")
        print("Preprocessing complete!")
        
        return X_test_scaled, y_test_encoded
    
    def get_feature_names(self):
        """Return list of feature names after preprocessing"""
        return self.feature_names
    
    def get_class_names(self):
        """Return list of class names"""
        return self.label_encoder.classes_