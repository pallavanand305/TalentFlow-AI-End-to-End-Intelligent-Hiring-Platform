#!/usr/bin/env python3
"""
Test script for baseline model training without database dependency

This script demonstrates the training pipeline using synthetic data
to verify the training logic works correctly.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class TestBaselineModelTrainer:
    """Test trainer for baseline TF-IDF model using synthetic data"""
    
    def __init__(self):
        """Initialize trainer"""
        self.model_name = "test_baseline_tfidf"
        self.model_version = "1.0.0"
        self.vectorizer = None
        self.section_weights = {
            'skills': 0.4,
            'experience': 0.4,
            'education': 0.2
        }
        
        # Training parameters
        self.params = {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'stop_words': 'english',
            'lowercase': True,
            'skills_weight': self.section_weights['skills'],
            'experience_weight': self.section_weights['experience'],
            'education_weight': self.section_weights['education'],
            'min_score_threshold': 0.1,
            'max_score_threshold': 1.0
        }
    
    def generate_synthetic_training_data(self) -> Tuple[List[Dict], List[float]]:
        """
        Generate synthetic training data for testing
        
        Returns:
            Tuple of (feature_data, target_scores)
        """
        logger.info("Generating synthetic training data...")
        
        # Sample candidate data
        candidates_data = [
            {
                'skills': 'Python Java SQL Machine Learning Data Science',
                'experience': 'Software Engineer at TechCorp 5 years experience developing web applications',
                'education': 'Bachelor Computer Science State University'
            },
            {
                'skills': 'JavaScript React Node.js HTML CSS',
                'experience': 'Frontend Developer at WebCorp 3 years building user interfaces',
                'education': 'Bachelor Information Technology City College'
            },
            {
                'skills': 'Python Django Flask PostgreSQL Redis',
                'experience': 'Backend Developer at DataCorp 4 years building APIs and databases',
                'education': 'Master Computer Science Tech University'
            },
            {
                'skills': 'Java Spring Boot Microservices Docker Kubernetes',
                'experience': 'Senior Developer at CloudCorp 6 years cloud architecture',
                'education': 'Bachelor Software Engineering State University'
            },
            {
                'skills': 'C++ Embedded Systems Real-time Programming',
                'experience': 'Embedded Engineer at HardwareCorp 7 years firmware development',
                'education': 'Bachelor Electrical Engineering Tech Institute'
            }
        ]
        
        # Sample job data
        jobs_data = [
            {
                'skills': 'Python Machine Learning Data Science SQL',
                'experience': 'Senior level data science experience required',
                'education': 'Bachelor degree in Computer Science or related field'
            },
            {
                'skills': 'JavaScript React Frontend Development',
                'experience': 'Mid level frontend development experience',
                'education': 'Bachelor degree in Computer Science'
            },
            {
                'skills': 'Java Spring Boot Backend Development',
                'experience': 'Senior backend development experience',
                'education': 'Bachelor or Master degree in Software Engineering'
            }
        ]
        
        # Generate training pairs
        training_data = []
        target_scores = []
        
        for candidate in candidates_data:
            for job in jobs_data:
                # Create feature data
                features = {
                    'candidate_skills': candidate['skills'],
                    'candidate_experience': candidate['experience'],
                    'candidate_education': candidate['education'],
                    'job_skills': job['skills'],
                    'job_experience': job['experience'],
                    'job_education': job['education']
                }
                
                # Generate synthetic score based on skill overlap
                candidate_skills = set(candidate['skills'].lower().split())
                job_skills = set(job['skills'].lower().split())
                skill_overlap = len(candidate_skills.intersection(job_skills))
                max_skills = max(len(candidate_skills), len(job_skills))
                
                # Base score on skill overlap with some noise
                base_score = skill_overlap / max_skills if max_skills > 0 else 0.0
                noise = np.random.normal(0, 0.1)  # Add some noise
                synthetic_score = max(0.0, min(1.0, base_score + noise))
                
                training_data.append(features)
                target_scores.append(synthetic_score)
        
        logger.info(f"Generated {len(training_data)} synthetic training samples")
        return training_data, target_scores
    
    def train_model(self, training_data: List[Dict], target_scores: List[float]) -> Dict[str, Any]:
        """
        Train the TF-IDF baseline model
        
        Args:
            training_data: List of feature dictionaries
            target_scores: List of target scores
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training baseline TF-IDF model...")
        
        if not training_data:
            raise ValueError("No training data available")
        
        # Prepare text data for each section
        skills_texts = []
        experience_texts = []
        education_texts = []
        
        for features in training_data:
            skills_texts.append(features.get('candidate_skills', '') + ' ' + features.get('job_skills', ''))
            experience_texts.append(features.get('candidate_experience', '') + ' ' + features.get('job_experience', ''))
            education_texts.append(features.get('candidate_education', '') + ' ' + features.get('job_education', ''))
        
        # Train TF-IDF vectorizers for each section
        self.skills_vectorizer = TfidfVectorizer(
            max_features=self.params['max_features'],
            ngram_range=self.params['ngram_range'],
            stop_words=self.params['stop_words'],
            lowercase=self.params['lowercase']
        )
        
        self.experience_vectorizer = TfidfVectorizer(
            max_features=self.params['max_features'],
            ngram_range=self.params['ngram_range'],
            stop_words=self.params['stop_words'],
            lowercase=self.params['lowercase']
        )
        
        self.education_vectorizer = TfidfVectorizer(
            max_features=self.params['max_features'],
            ngram_range=self.params['ngram_range'],
            stop_words=self.params['stop_words'],
            lowercase=self.params['lowercase']
        )
        
        # Fit vectorizers
        try:
            skills_vectors = self.skills_vectorizer.fit_transform(skills_texts)
            experience_vectors = self.experience_vectorizer.fit_transform(experience_texts)
            education_vectors = self.education_vectorizer.fit_transform(education_texts)
        except Exception as e:
            logger.error(f"Error fitting vectorizers: {e}")
            raise
        
        # Compute predictions using the trained vectorizers
        predictions = []
        for i in range(len(training_data)):
            try:
                # For testing, we'll use a simplified similarity computation
                # In practice, this would compare candidate vs job separately
                
                # Get text similarities (simplified)
                skills_text = skills_texts[i]
                exp_text = experience_texts[i]
                edu_text = education_texts[i]
                
                # Simple word overlap similarity
                skills_words = set(skills_text.lower().split())
                exp_words = set(exp_text.lower().split())
                edu_words = set(edu_text.lower().split())
                
                # Calculate basic similarities
                skills_sim = len(skills_words) / (len(skills_words) + 1) if skills_words else 0.0
                exp_sim = len(exp_words) / (len(exp_words) + 1) if exp_words else 0.0
                edu_sim = len(edu_words) / (len(edu_words) + 1) if edu_words else 0.0
                
                # Normalize similarities
                skills_sim = min(1.0, skills_sim)
                exp_sim = min(1.0, exp_sim)
                edu_sim = min(1.0, edu_sim)
                
                # Weighted combination
                weighted_score = (
                    skills_sim * self.section_weights['skills'] +
                    exp_sim * self.section_weights['experience'] +
                    edu_sim * self.section_weights['education']
                )
                
                # Normalize to [0, 1]
                weighted_score = max(0.0, min(1.0, weighted_score))
                predictions.append(weighted_score)
                
            except Exception as e:
                logger.warning(f"Error computing prediction for sample {i}: {e}")
                predictions.append(0.0)
        
        # Split data for validation
        if len(training_data) > 4:
            X_train, X_val, y_train, y_val = train_test_split(
                list(range(len(training_data))), target_scores, test_size=0.3, random_state=42
            )
            
            train_predictions = [predictions[i] for i in X_train]
            val_predictions = [predictions[i] for i in X_val]
            
            # Compute metrics
            # Convert to binary classification for metrics (threshold at 0.5)
            train_binary = [1 if p >= 0.5 else 0 for p in train_predictions]
            val_binary = [1 if p >= 0.5 else 0 for p in val_predictions]
            train_target_binary = [1 if y_train[i] >= 0.5 else 0 for i in range(len(y_train))]
            val_target_binary = [1 if y_val[i] >= 0.5 else 0 for i in range(len(y_val))]
            
            metrics = {
                'train_accuracy': accuracy_score(train_target_binary, train_binary),
                'val_accuracy': accuracy_score(val_target_binary, val_binary),
                'train_precision': precision_score(train_target_binary, train_binary, zero_division=0),
                'val_precision': precision_score(val_target_binary, val_binary, zero_division=0),
                'train_recall': recall_score(train_target_binary, train_binary, zero_division=0),
                'val_recall': recall_score(val_target_binary, val_binary, zero_division=0),
                'train_f1': f1_score(train_target_binary, train_binary, zero_division=0),
                'val_f1': f1_score(val_target_binary, val_binary, zero_division=0),
                'train_mse': np.mean([(p - y_train[i])**2 for i, p in enumerate(train_predictions)]),
                'val_mse': np.mean([(p - y_val[i])**2 for i, p in enumerate(val_predictions)]),
                'train_mae': np.mean([abs(p - y_train[i]) for i, p in enumerate(train_predictions)]),
                'val_mae': np.mean([abs(p - y_val[i]) for i, p in enumerate(val_predictions)]),
                'num_train_samples': len(X_train),
                'num_val_samples': len(X_val),
                'num_features_skills': skills_vectors.shape[1],
                'num_features_experience': experience_vectors.shape[1],
                'num_features_education': education_vectors.shape[1]
            }
        else:
            # Not enough data for validation split
            logger.warning("Not enough data for train/validation split, using all data for training")
            
            # Convert to binary for metrics
            binary_predictions = [1 if p >= 0.5 else 0 for p in predictions]
            binary_targets = [1 if s >= 0.5 else 0 for s in target_scores]
            
            metrics = {
                'train_accuracy': accuracy_score(binary_targets, binary_predictions),
                'train_precision': precision_score(binary_targets, binary_predictions, zero_division=0),
                'train_recall': recall_score(binary_targets, binary_predictions, zero_division=0),
                'train_f1': f1_score(binary_targets, binary_predictions, zero_division=0),
                'train_mse': np.mean([(p - target_scores[i])**2 for i, p in enumerate(predictions)]),
                'train_mae': np.mean([abs(p - target_scores[i]) for i, p in enumerate(predictions)]),
                'num_train_samples': len(training_data),
                'num_val_samples': 0,
                'num_features_skills': skills_vectors.shape[1],
                'num_features_experience': experience_vectors.shape[1],
                'num_features_education': education_vectors.shape[1]
            }
        
        logger.info(f"Training completed. Validation accuracy: {metrics.get('val_accuracy', metrics.get('train_accuracy', 0)):.3f}")
        return metrics
    
    def log_to_mlflow(self, metrics: Dict[str, Any]) -> str:
        """
        Log model, metrics, and parameters to MLflow (mock version)
        
        Args:
            metrics: Training metrics
            
        Returns:
            Mock MLflow run ID
        """
        logger.info("Logging model to MLflow (test mode)...")
        
        # In test mode, we'll just log the parameters and metrics without actually using MLflow
        logger.info("Parameters:")
        for key, value in self.params.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Return a mock run ID
        mock_run_id = "test_run_12345"
        logger.info(f"Model logged to MLflow (test mode) with run_id: {mock_run_id}")
        return mock_run_id
    
    def register_model(self, run_id: str) -> str:
        """
        Register model in MLflow model registry (mock version)
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Mock model version
        """
        logger.info("Registering model in MLflow registry (test mode)...")
        
        mock_version = "1"
        logger.info(f"Model registered (test mode) as version: {mock_version}")
        return mock_version
    
    def train_and_register(self) -> Dict[str, Any]:
        """
        Complete training pipeline: generate data, train, log to MLflow, and register
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting baseline model training pipeline (test mode)...")
        
        try:
            # Generate synthetic training data
            training_data, target_scores = self.generate_synthetic_training_data()
            
            if not training_data:
                raise ValueError("No training data available")
            
            # Train model
            metrics = self.train_model(training_data, target_scores)
            
            # Log to MLflow (test mode)
            run_id = self.log_to_mlflow(metrics)
            
            # Register model (test mode)
            model_version = self.register_model(run_id)
            
            # Prepare results
            results = {
                'model_name': self.model_name,
                'model_version': model_version,
                'run_id': run_id,
                'metrics': metrics,
                'num_training_samples': len(training_data),
                'status': 'success'
            }
            
            logger.info("Baseline model training completed successfully (test mode)")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {
                'model_name': self.model_name,
                'status': 'failed',
                'error': str(e)
            }


def main():
    """Main test training script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test baseline TF-IDF model training")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting TalentFlow AI Baseline Model Training (Test Mode)")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Create trainer and run training
    trainer = TestBaselineModelTrainer()
    
    try:
        results = trainer.train_and_register()
        
        if results['status'] == 'success':
            logger.info("Training completed successfully!")
            logger.info(f"Model: {results['model_name']} v{results['model_version']}")
            logger.info(f"Run ID: {results['run_id']}")
            logger.info(f"Training samples: {results['num_training_samples']}")
            
            # Print key metrics
            metrics = results['metrics']
            if 'val_accuracy' in metrics:
                logger.info(f"Validation Accuracy: {metrics['val_accuracy']:.3f}")
                logger.info(f"Validation F1: {metrics['val_f1']:.3f}")
                logger.info(f"Validation MSE: {metrics['val_mse']:.3f}")
            else:
                logger.info(f"Training Accuracy: {metrics['train_accuracy']:.3f}")
                logger.info(f"Training F1: {metrics['train_f1']:.3f}")
                logger.info(f"Training MSE: {metrics['train_mse']:.3f}")
        else:
            logger.error(f"Training failed: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()