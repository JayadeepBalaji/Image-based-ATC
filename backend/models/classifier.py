import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from ..utils.serialization import to_py

class AnimalClassifier:
    """
    Machine Learning classifier for animal productivity assessment.
    Uses Random Forest to classify animals into High/Medium/Low productivity categories.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.loaded = False
        self.model_path = "../data/animal_classifier_model.pkl"
        self.scaler_path = "../data/feature_scaler.pkl"
        self.metadata_path = "../data/model_metadata.json"
        
        # Ensure data directory exists
        os.makedirs("../data", exist_ok=True)
        
        # Define feature importance weights for scoring
        self.feature_weights = {
            'body_length_ratio': 0.20,
            'chest_depth_ratio': 0.18,
            'leg_length_ratio': 0.15,
            'body_condition_score': 0.22,
            'structural_soundness': 0.25
        }
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.loaded
    
    def generate_training_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate simulated training data for the classifier.
        
        Args:
            n_samples: Number of training samples to generate
            
        Returns:
            Tuple of (features DataFrame, labels array)
        """
        np.random.seed(42)  # For reproducible results
        
        data = []
        labels = []
        
        for i in range(n_samples):
            # Generate features for different productivity levels
            productivity_level = np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2])
            species = np.random.choice(['cow', 'buffalo'])
            
            if productivity_level == 'High':
                # High productivity animals - optimal measurements
                body_length_ratio = np.random.normal(1.4, 0.1)
                chest_depth_ratio = np.random.normal(0.75, 0.05)
                leg_length_ratio = np.random.normal(0.5, 0.03)
                body_condition_score = np.random.normal(0.85, 0.1)
                structural_soundness = np.random.normal(0.9, 0.05)
                udder_placement_score = np.random.normal(0.8, 0.1)
                
                # Productivity indicators
                body_proportions = np.random.normal(0.85, 0.08)
                chest_capacity = np.random.normal(0.8, 0.08)
                leg_structure = np.random.normal(0.85, 0.06)
                overall_conformation = np.random.normal(0.88, 0.05)
                
            elif productivity_level == 'Medium':
                # Medium productivity animals - average measurements
                body_length_ratio = np.random.normal(1.35, 0.15)
                chest_depth_ratio = np.random.normal(0.7, 0.08)
                leg_length_ratio = np.random.normal(0.48, 0.05)
                body_condition_score = np.random.normal(0.65, 0.15)
                structural_soundness = np.random.normal(0.7, 0.1)
                udder_placement_score = np.random.normal(0.65, 0.15)
                
                # Productivity indicators
                body_proportions = np.random.normal(0.65, 0.12)
                chest_capacity = np.random.normal(0.6, 0.12)
                leg_structure = np.random.normal(0.65, 0.1)
                overall_conformation = np.random.normal(0.65, 0.1)
                
            else:  # Low productivity
                # Low productivity animals - suboptimal measurements
                body_length_ratio = np.random.normal(1.2, 0.2)
                chest_depth_ratio = np.random.normal(0.6, 0.1)
                leg_length_ratio = np.random.normal(0.45, 0.08)
                body_condition_score = np.random.normal(0.4, 0.15)
                structural_soundness = np.random.normal(0.45, 0.15)
                udder_placement_score = np.random.normal(0.4, 0.15)
                
                # Productivity indicators
                body_proportions = np.random.normal(0.4, 0.15)
                chest_capacity = np.random.normal(0.35, 0.15)
                leg_structure = np.random.normal(0.4, 0.15)
                overall_conformation = np.random.normal(0.4, 0.12)
            
            # Add some species-specific variations
            if species == 'buffalo':
                body_length_ratio *= 0.95  # Slightly more compact
                chest_depth_ratio *= 1.05  # Broader chest
            
            # Ensure values are within reasonable bounds
            body_length_ratio = np.clip(body_length_ratio, 0.8, 2.0)
            chest_depth_ratio = np.clip(chest_depth_ratio, 0.3, 1.2)
            leg_length_ratio = np.clip(leg_length_ratio, 0.2, 0.8)
            body_condition_score = np.clip(body_condition_score, 0.0, 1.0)
            structural_soundness = np.clip(structural_soundness, 0.0, 1.0)
            udder_placement_score = np.clip(udder_placement_score, 0.0, 1.0)
            body_proportions = np.clip(body_proportions, 0.0, 1.0)
            chest_capacity = np.clip(chest_capacity, 0.0, 1.0)
            leg_structure = np.clip(leg_structure, 0.0, 1.0)
            overall_conformation = np.clip(overall_conformation, 0.0, 1.0)
            
            # Create feature vector
            features = {
                'species_cow': 1 if species == 'cow' else 0,
                'species_buffalo': 1 if species == 'buffalo' else 0,
                'body_length_ratio': body_length_ratio,
                'chest_depth_ratio': chest_depth_ratio,
                'leg_length_ratio': leg_length_ratio,
                'body_condition_score': body_condition_score,
                'structural_soundness': structural_soundness,
                'udder_placement_score': udder_placement_score,
                'body_proportions': body_proportions,
                'chest_capacity': chest_capacity,
                'leg_structure': leg_structure,
                'overall_conformation': overall_conformation
            }
            
            data.append(features)
            labels.append(productivity_level)
        
        # Convert to DataFrame and array
        df = pd.DataFrame(data)
        labels = np.array(labels)
        
        return df, labels
    
    def train_model(self, retrain: bool = False):
        """
        Train the Random Forest classifier.
        
        Args:
            retrain: Whether to retrain even if model exists
        """
        try:
            # Check if model already exists and don't retrain unless specified
            if os.path.exists(self.model_path) and not retrain:
                print("Model already exists. Use retrain=True to force retraining.")
                return
            
            print("Generating training data...")
            X, y = self.generate_training_data(n_samples=2000)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest
            print("Training Random Forest model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Model trained successfully!")
            print(f"Accuracy: {accuracy:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Save feature names
            self.feature_names = list(X.columns)
            
            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'accuracy': float(accuracy),
                'training_date': datetime.now().isoformat(),
                'n_samples': len(X),
                'classes': list(self.model.classes_)
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.loaded = True
            print("Model saved successfully!")
            
        except Exception as e:
            print(f"Error training model: {e}")
            self.loaded = False
    
    def load_model(self):
        """Load the trained model from disk."""
        try:
            if not os.path.exists(self.model_path):
                print("Model not found. Training new model...")
                self.train_model()
                return
            
            # Load model and scaler
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
            
            self.loaded = True
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.loaded = False
    
    def prepare_features(self, features: Dict, species: str) -> Optional[np.ndarray]:
        """
        Prepare features for prediction.
        
        Args:
            features: Dictionary of extracted features
            species: Animal species
            
        Returns:
            Prepared feature array
        """
        try:
            # Create feature vector matching training data
            feature_vector = {
                'species_cow': 1 if species == 'cow' else 0,
                'species_buffalo': 1 if species == 'buffalo' else 0,
                'body_length_ratio': features.get('body_length_ratio', 0),
                'chest_depth_ratio': features.get('chest_depth_ratio', 0),
                'leg_length_ratio': features.get('leg_length_ratio', 0),
                'body_condition_score': features.get('body_condition_score', 0),
                'structural_soundness': features.get('structural_soundness', 0),
                'udder_placement_score': features.get('udder_placement_score', 0),
                'body_proportions': features.get('productivity_indicators', {}).get('body_proportions', 0),
                'chest_capacity': features.get('productivity_indicators', {}).get('chest_capacity', 0),
                'leg_structure': features.get('productivity_indicators', {}).get('leg_structure', 0),
                'overall_conformation': features.get('productivity_indicators', {}).get('overall_conformation', 0)
            }
            
            # Convert to DataFrame with correct column order
            df = pd.DataFrame([feature_vector])
            if self.feature_names:
                df = df[self.feature_names]
            
            return df.values
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def calculate_trait_scores(self, features: Dict) -> Dict:
        """
        Calculate individual trait scores for detailed feedback.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary of trait scores
        """
        try:
            scores = {}
            
            # Body structure scores
            scores['body_length'] = {
                'value': features.get('body_length_ratio', 0),
                'score': min(100, max(0, features.get('body_length_ratio', 0) * 70)),
                'ideal_range': '1.3 - 1.5',
                'description': 'Optimal body length indicates good growth potential'
            }
            
            scores['chest_depth'] = {
                'value': features.get('chest_depth_ratio', 0),
                'score': min(100, max(0, features.get('chest_depth_ratio', 0) * 130)),
                'ideal_range': '0.65 - 0.80',
                'description': 'Good chest depth indicates lung and heart capacity'
            }
            
            scores['leg_structure'] = {
                'value': features.get('leg_length_ratio', 0),
                'score': min(100, max(0, features.get('leg_length_ratio', 0) * 200)),
                'ideal_range': '0.45 - 0.55',
                'description': 'Proper leg structure ensures mobility and longevity'
            }
            
            scores['body_condition'] = {
                'value': features.get('body_condition_score', 0),
                'score': features.get('body_condition_score', 0) * 100,
                'ideal_range': '0.7 - 0.9',
                'description': 'Good body condition indicates overall health'
            }
            
            scores['structural_soundness'] = {
                'value': features.get('structural_soundness', 0),
                'score': features.get('structural_soundness', 0) * 100,
                'ideal_range': '0.8 - 1.0',
                'description': 'Structural soundness affects productivity and lifespan'
            }
            
            scores['udder_placement'] = {
                'value': features.get('udder_placement_score', 0),
                'score': features.get('udder_placement_score', 0) * 100,
                'ideal_range': '0.7 - 0.9',
                'description': 'Good udder placement improves milking efficiency'
            }
            
            return scores
            
        except Exception as e:
            print(f"Error calculating trait scores: {e}")
            return {}
    
    def classify(self, features: Dict, species: str) -> Dict:
        """
        Classify an animal based on extracted features.
        
        Args:
            features: Dictionary of extracted features
            species: Animal species
            
        Returns:
            Classification result with details
        """
        try:
            if not self.loaded:
                return {
                    'error': 'Model not loaded',
                    'productivity_class': 'Unknown',
                    'confidence': 0.0
                }
            
            # Prepare features
            feature_array = self.prepare_features(features, species)
            if feature_array is None:
                return {
                    'error': 'Could not prepare features',
                    'productivity_class': 'Unknown',
                    'confidence': 0.0
                }
            
            # Scale features
            assert self.scaler is not None
            feature_array_scaled = self.scaler.transform(feature_array)
            
            # Make prediction
            assert self.model is not None
            prediction = self.model.predict(feature_array_scaled)[0]
            probabilities = self.model.predict_proba(feature_array_scaled)[0]
            
            # Get confidence (highest probability)
            confidence = float(max(probabilities))
            
            # Create probability distribution
            prob_dict = {}
            for i, class_name in enumerate(self.model.classes_):
                prob_dict[str(class_name)] = float(probabilities[i])
            
            # Calculate trait scores
            trait_scores = self.calculate_trait_scores(features)
            
            # Calculate overall score
            overall_score = sum(float(score['score']) for score in trait_scores.values()) / len(trait_scores) if trait_scores else 0
            
            # Generate recommendations
            recommendations = self.generate_recommendations(trait_scores, prediction)
            
            result = {
                'productivity_class': str(prediction),
                'confidence': float(confidence),
                'probability_distribution': prob_dict,
                'overall_score': round(float(overall_score), 1),
                'trait_scores': to_py(trait_scores),
                'recommendations': [str(r) for r in recommendations],
                'success': True
            }
            
            return result
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return {
                'error': str(e),
                'productivity_class': 'Unknown',
                'confidence': 0.0,
                'success': False
            }
    
    def generate_recommendations(self, trait_scores: Dict, productivity_class: str) -> List[str]:
        """
        Generate recommendations based on trait scores and classification.
        
        Args:
            trait_scores: Dictionary of individual trait scores
            productivity_class: Predicted productivity class
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            if productivity_class == 'High':
                recommendations.append("Excellent animal with high productivity potential")
                recommendations.append("Maintain good nutrition and regular health checks")
                recommendations.append("Consider for breeding programs")
            
            elif productivity_class == 'Medium':
                recommendations.append("Good animal with moderate productivity potential")
                
                # Check specific areas for improvement
                if trait_scores.get('body_condition', {}).get('score', 0) < 60:
                    recommendations.append("Improve nutrition to enhance body condition")
                
                if trait_scores.get('chest_depth', {}).get('score', 0) < 60:
                    recommendations.append("Monitor respiratory health and feed quality")
                
                if trait_scores.get('leg_structure', {}).get('score', 0) < 60:
                    recommendations.append("Provide proper flooring and hoof care")
                
                recommendations.append("Regular health monitoring recommended")
            
            else:  # Low productivity
                recommendations.append("Animal shows signs of productivity limitations")
                recommendations.append("Comprehensive health evaluation recommended")
                recommendations.append("Consider management improvements:")
                
                if trait_scores.get('body_condition', {}).get('score', 0) < 40:
                    recommendations.append("- Nutritional supplementation needed")
                
                if trait_scores.get('structural_soundness', {}).get('score', 0) < 40:
                    recommendations.append("- Address structural issues")
                
                recommendations.append("May not be suitable for breeding")
            
            # Add general recommendations
            if not recommendations:
                recommendations.append("Maintain regular health monitoring")
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return ["Regular health monitoring recommended"]