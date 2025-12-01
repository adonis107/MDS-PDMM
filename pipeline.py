import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, confusion_matrix
import matplotlib.pyplot as plt
import preprocessing as prep
import machine_learning as ml

class AnomalyDetectionPipeline:
    def __init__(self, seq_length=25, batch_size=64, device=None,random_state=0):
        
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State variables
        self.raw_df = None
        self.processed_df = None
        self.feature_names = []
        self.scaler = None
        self.model = None
        self.detector = None # e.g., OC-SVM
        self.model_type = None
        
        print(f"Pipeline initialized on device: {self.device}")

    def load_data(self, filepath):
        """Loads data from CSV/Parquet."""
        print(f"Loading data from {filepath}...")

        if filepath.endswith('.csv') or filepath.endswith('.csv.gz'):
            self.raw_df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            self.raw_df = pd.read_parquet(filepath)
        else:
            raise ValueError("Unsupported file format")
        
        return self

    def engineer_features(self, feature_sets=['base', 'tao', 'poutre', 'hawkes', 'slopes']):
        """
        Applies feature engineering based on selected sets.
        Options: 'base' (Basic LOB), 'tao' (Weighted Imbalance), 
                 'poutre' (Rapidity), 'hawkes' (Memory), 'slopes' (Elasticity)
        """
        print(f"Engineering features: {feature_sets}...")

        df = self.raw_df.copy()
        
        # Base extraction
        features = prep.extract_features(df, window=50)
        
        if 'tao' in feature_sets:
            features["Weighted_Imbalance_decreasing"] = prep.compute_weighted_imbalance(df, weights=[0.1, 0.1, 0.2, 0.2, 0.4], levels=5)
            features["Weighted_Imbalance_increasing"] = prep.compute_weighted_imbalance(df, weights=[0.4, 0.2, 0.2, 0.1, 0.1], levels=5)
            features["Weighted_Imbalance_constant"] = prep.compute_weighted_imbalance(df, weights=[0.2, 0.2, 0.2, 0.2, 0.2], levels=5)
        
        if 'poutre' in feature_sets:
            features = prep.compute_rapidity_event_flow_features(df, features)
            
        if 'hawkes' in feature_sets:
            features = prep.compute_hawkes_and_weighted_flow(df, data=features)

        # Cleanup
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features = features.fillna(0)
        
        # Clip extreme outliers
        lower = features.quantile(0.001)
        upper = features.quantile(0.999)
        features = features.clip(lower=lower, upper=upper, axis=1)
        
        self.processed_df = features
        self.feature_names = features.columns.tolist()
        print(f"Feature Engineering complete. Total features: {len(self.feature_names)}")
        return self

    def scale_and_sequence(self, method='minmax', train_split=0.7):
        """
        Scales data and creates sequences.
        method: 'minmax' (default), 'standard', 'box-cox'
        """
        print(f"Preprocessing with method: {method}...")

        data_values = self.processed_df.values
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
            data_scaled = self.scaler.fit_transform(data_values)
        elif method == 'standard':
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data_values)
        elif method == 'box-cox':
            self.scaler = prep.data_preprocessor() # Uses the class from preprocessing.py
            data_scaled = self.scaler.fit_transform(data_values)
        else:
            raise ValueError(f"Unknown scaler method: {method}")

        # Sequence Generation
        # Note: PNN usually takes 2D input (Event t), Transformer takes 3D (Sequence t-k..t)
        # For uniformity in this wrapper, we generate sequences. 
        # If PNN is used, we might flatten or take the last step later.
        sequences = prep.create_sequences(data_scaled, self.seq_length)
        
        # Split
        train_size = int(len(sequences) * train_split)
        self.X_train = sequences[:train_size]
        self.X_test = sequences[train_size:]
        
        print(f"Data split: Train {self.X_train.shape}, Test {self.X_test.shape}")
        return self

    def _get_dataloader(self, X, shuffle=True):
        tensor_x = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_x) # Autoencoder target is input
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_model(self, model_type='transformer_ocsvm', epochs=5, lr=1e-3, nu=0.01):
        """
        Trains the selected model architecture.
        model_type: 'transformer_ocsvm' (default), 'pnn' (future implementation)
        """
        self.model_type = model_type
        num_feat = self.X_train.shape[2]
        
        if model_type == 'transformer_ocsvm':
            print("Initializing Transformer Autoencoder...")
            self.model = ml.TransformerAutoencoder(
                num_features=num_feat,
                model_dim=64,
                num_heads=4,
                num_layers=2,
                representation_dim=128,
                sequence_length=self.seq_length
            ).to(self.device)
            
            # Train Autoencoder
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            train_loader = self._get_dataloader(self.X_train)
            
            self.model.train()
            print("Training Autoencoder...")
            for epoch in range(epochs):
                total_loss = 0
                for batch_data, _ in train_loader:
                    batch_data = batch_data.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(batch_data)
                    loss = criterion(output, batch_data)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.6f}")
            
            # Extract Latent Representations
            print("Extracting Latent Representations for OC-SVM...")
            z_train = self._get_latent(self.X_train)
            
            # Scale Latent Space (Important for SVM convergence)
            self.latent_scaler = StandardScaler()
            z_train_scaled = self.latent_scaler.fit_transform(z_train)
            
            # Train OC-SVM
            print(f"Training One-Class SVM (nu={nu})...")
            self.detector = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)
            self.detector.fit(z_train_scaled)
            
        elif model_type == 'pnn':
            raise NotImplementedError("PNN integration coming soon.")
            
        return self

    def _get_latent(self, X):
        """Helper to get embeddings in batches"""

        loader = self._get_dataloader(X, shuffle=False)
        self.model.eval()
        reps = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device)
                # Ensure model has get_representation method
                r = self.model.get_representation(inputs)
                reps.append(r.cpu().numpy())

        return np.concatenate(reps, axis=0)

    def evaluate(self, y_true=None):
        """
        Evaluates the model.
        For unsupervised, we create synthetic anomalies if y_true is None.
        """
        print("Evaluating model...")
        
        # Get Normal Test Data Representations
        z_test_normal = self._get_latent(self.X_test)
        z_test_normal = self.latent_scaler.transform(z_test_normal)
        
        if y_true is None:
            # Synthetic Anomalies (Scale Factor perturbation)
            # In a real pipeline, you might want specific attack simulations here
            z_test_anom = z_test_normal * 5.0

            # Create synthetic test set
            X_eval = np.concatenate([z_test_normal, z_test_anom], axis=0)
            y_eval = np.concatenate([np.zeros(len(z_test_normal)), np.ones(len(z_test_anom))])
        else:
            X_eval = z_test_normal
            y_eval = y_true

        # Score
        scores = -self.detector.score_samples(X_eval) # Higher = more anomalous
        preds = self.detector.predict(X_eval)
        preds = np.where(preds == -1, 1, 0)
        
        # Metrics
        results = {
            "AUROC": roc_auc_score(y_eval, scores),
            "AUPRC": average_precision_score(y_eval, scores),
            "F4_Score": fbeta_score(y_eval, preds, beta=4)
        }
        cm = confusion_matrix(y_eval, preds)
        
        return results, cm

    def get_feature_importance(self, n_repeats=3):
        """
        Calculates permutation importance on the Test set (Normal data).
        """
        print("Calculating Feature Importance (Permutation)...")

        self.model.eval()
        
        # Use a subset for speed
        subset_idx = np.random.choice(len(self.X_test), size=min(1000, len(self.X_test)), replace=False)
        X_subset = self.X_test[subset_idx]
        
        # Baseline score (mean anomaly score of normal data)
        z_base = self._get_latent(X_subset)
        z_base = self.latent_scaler.transform(z_base)
        base_scores = -self.detector.score_samples(z_base)
        base_mean = np.mean(base_scores)
        
        importances = []
        
        tensor_subset = torch.tensor(X_subset, dtype=torch.float32).to(self.device)
        
        for i, name in enumerate(self.feature_names):
            diffs = []
            for _ in range(n_repeats):
                # Permute feature i
                permuted = tensor_subset.clone()
                idx = torch.randperm(permuted.size(0))
                permuted[:, :, i] = permuted[idx, :, i]
                
                # Forward pass
                with torch.no_grad():
                    z_perm = self.model.get_representation(permuted).cpu().numpy()
                
                z_perm = self.latent_scaler.transform(z_perm)
                perm_scores = -self.detector.score_samples(z_perm)
                
                # Impact: How much did the anomaly score deviate from baseline?
                # We expect shuffling a key feature to make normal data look anomalous
                diff = np.mean(np.abs(perm_scores - base_scores))
                diffs.append(diff)
            
            importances.append(np.mean(diffs))
            
        imp_df = pd.DataFrame({'Feature': self.feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        
        # Normalize
        imp_df['Importance'] = (imp_df['Importance'] / imp_df['Importance'].max()) * 100
        
        return imp_df