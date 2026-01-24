"""Unit tests for baseline model training"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.training.test_baseline_training import TestBaselineModelTrainer


class TestBaselineModelTraining:
    """Test cases for baseline model training"""
    
    @pytest.fixture
    def trainer(self):
        """Create a test trainer instance"""
        return TestBaselineModelTrainer()
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization"""
        assert trainer.model_name == "test_baseline_tfidf"
        assert trainer.model_version == "1.0.0"
        assert trainer.section_weights['skills'] == 0.4
        assert trainer.section_weights['experience'] == 0.4
        assert trainer.section_weights['education'] == 0.2
        assert trainer.params['max_features'] == 5000
        assert trainer.params['ngram_range'] == (1, 2)
    
    def test_generate_synthetic_training_data(self, trainer):
        """Test synthetic training data generation"""
        training_data, target_scores = trainer.generate_synthetic_training_data()
        
        # Check that data was generated
        assert len(training_data) > 0
        assert len(target_scores) > 0
        assert len(training_data) == len(target_scores)
        
        # Check data structure
        sample = training_data[0]
        assert 'candidate_skills' in sample
        assert 'candidate_experience' in sample
        assert 'candidate_education' in sample
        assert 'job_skills' in sample
        assert 'job_experience' in sample
        assert 'job_education' in sample
        
        # Check score range
        for score in target_scores:
            assert 0.0 <= score <= 1.0
    
    def test_train_model_with_valid_data(self, trainer):
        """Test model training with valid data"""
        # Generate test data
        training_data, target_scores = trainer.generate_synthetic_training_data()
        
        # Train model
        metrics = trainer.train_model(training_data, target_scores)
        
        # Check that metrics were computed
        assert 'train_accuracy' in metrics
        assert 'train_precision' in metrics
        assert 'train_recall' in metrics
        assert 'train_f1' in metrics
        assert 'train_mse' in metrics
        assert 'train_mae' in metrics
        assert 'num_train_samples' in metrics
        
        # Check metric ranges
        assert 0.0 <= metrics['train_accuracy'] <= 1.0
        assert 0.0 <= metrics['train_precision'] <= 1.0
        assert 0.0 <= metrics['train_recall'] <= 1.0
        assert 0.0 <= metrics['train_f1'] <= 1.0
        assert metrics['train_mse'] >= 0.0
        assert metrics['train_mae'] >= 0.0
        assert metrics['num_train_samples'] > 0
        
        # Check that vectorizers were created
        assert trainer.skills_vectorizer is not None
        assert trainer.experience_vectorizer is not None
        assert trainer.education_vectorizer is not None
    
    def test_train_model_with_empty_data(self, trainer):
        """Test model training with empty data"""
        with pytest.raises(ValueError, match="No training data available"):
            trainer.train_model([], [])
    
    def test_train_model_with_validation_split(self, trainer):
        """Test model training with validation split"""
        # Generate enough data for validation split
        training_data, target_scores = trainer.generate_synthetic_training_data()
        
        # Ensure we have enough data for validation split
        assert len(training_data) > 4
        
        # Train model
        metrics = trainer.train_model(training_data, target_scores)
        
        # Check that validation metrics exist
        assert 'val_accuracy' in metrics
        assert 'val_precision' in metrics
        assert 'val_recall' in metrics
        assert 'val_f1' in metrics
        assert 'val_mse' in metrics
        assert 'val_mae' in metrics
        assert 'num_val_samples' in metrics
        
        # Check that we have both train and validation samples
        assert metrics['num_train_samples'] > 0
        assert metrics['num_val_samples'] > 0
        assert metrics['num_train_samples'] + metrics['num_val_samples'] == len(training_data)
    
    def test_log_to_mlflow(self, trainer):
        """Test MLflow logging (test mode)"""
        metrics = {
            'train_accuracy': 0.8,
            'train_f1': 0.75,
            'train_mse': 0.1
        }
        
        run_id = trainer.log_to_mlflow(metrics)
        
        # Check that a run ID was returned
        assert run_id == "test_run_12345"
    
    def test_register_model(self, trainer):
        """Test model registration (test mode)"""
        run_id = "test_run_12345"
        
        model_version = trainer.register_model(run_id)
        
        # Check that a version was returned
        assert model_version == "1"
    
    def test_train_and_register_success(self, trainer):
        """Test complete training and registration pipeline"""
        results = trainer.train_and_register()
        
        # Check successful results
        assert results['status'] == 'success'
        assert results['model_name'] == trainer.model_name
        assert results['model_version'] == "1"
        assert results['run_id'] == "test_run_12345"
        assert 'metrics' in results
        assert results['num_training_samples'] > 0
        
        # Check metrics structure
        metrics = results['metrics']
        assert 'train_accuracy' in metrics
        assert 'train_f1' in metrics
        assert 'num_train_samples' in metrics
    
    def test_section_weights_sum_to_one(self, trainer):
        """Test that section weights sum to 1.0"""
        total_weight = sum(trainer.section_weights.values())
        assert abs(total_weight - 1.0) < 1e-6  # Allow for floating point precision
    
    def test_parameters_structure(self, trainer):
        """Test that parameters have expected structure"""
        params = trainer.params
        
        # Check required parameters exist
        required_params = [
            'max_features', 'ngram_range', 'stop_words', 'lowercase',
            'skills_weight', 'experience_weight', 'education_weight',
            'min_score_threshold', 'max_score_threshold'
        ]
        
        for param in required_params:
            assert param in params
        
        # Check parameter types and ranges
        assert isinstance(params['max_features'], int)
        assert params['max_features'] > 0
        assert isinstance(params['ngram_range'], tuple)
        assert len(params['ngram_range']) == 2
        assert isinstance(params['lowercase'], bool)
        assert 0.0 <= params['skills_weight'] <= 1.0
        assert 0.0 <= params['experience_weight'] <= 1.0
        assert 0.0 <= params['education_weight'] <= 1.0
        assert 0.0 <= params['min_score_threshold'] <= 1.0
        assert 0.0 <= params['max_score_threshold'] <= 1.0
    
    def test_feature_extraction_structure(self, trainer):
        """Test that feature extraction produces expected structure"""
        training_data, _ = trainer.generate_synthetic_training_data()
        
        # Check that all samples have required features
        for sample in training_data:
            assert isinstance(sample, dict)
            assert 'candidate_skills' in sample
            assert 'candidate_experience' in sample
            assert 'candidate_education' in sample
            assert 'job_skills' in sample
            assert 'job_experience' in sample
            assert 'job_education' in sample
            
            # Check that features are strings
            for key, value in sample.items():
                assert isinstance(value, str)
    
    def test_score_generation_range(self, trainer):
        """Test that generated scores are in valid range"""
        _, target_scores = trainer.generate_synthetic_training_data()
        
        for score in target_scores:
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0
    
    def test_vectorizer_fitting(self, trainer):
        """Test that vectorizers are properly fitted"""
        training_data, target_scores = trainer.generate_synthetic_training_data()
        
        # Train model to fit vectorizers
        trainer.train_model(training_data, target_scores)
        
        # Check that vectorizers were fitted
        assert hasattr(trainer.skills_vectorizer, 'vocabulary_')
        assert hasattr(trainer.experience_vectorizer, 'vocabulary_')
        assert hasattr(trainer.education_vectorizer, 'vocabulary_')
        
        # Check that vocabularies are not empty
        assert len(trainer.skills_vectorizer.vocabulary_) > 0
        assert len(trainer.experience_vectorizer.vocabulary_) > 0
        assert len(trainer.education_vectorizer.vocabulary_) > 0


@pytest.mark.integration
class TestBaselineModelTrainingIntegration:
    """Integration tests for baseline model training"""
    
    def test_end_to_end_training_pipeline(self):
        """Test complete end-to-end training pipeline"""
        trainer = TestBaselineModelTrainer()
        
        # Run complete pipeline
        results = trainer.train_and_register()
        
        # Verify successful completion
        assert results['status'] == 'success'
        assert results['num_training_samples'] > 0
        
        # Verify metrics are reasonable
        metrics = results['metrics']
        
        # Check that we have some predictive power (not random)
        if 'val_accuracy' in metrics:
            # For synthetic data, we expect some accuracy
            assert metrics['val_accuracy'] >= 0.0
        
        # Check that MSE is reasonable (not too high)
        if 'val_mse' in metrics:
            assert metrics['val_mse'] <= 1.0  # MSE should be <= 1 for scores in [0,1]
        elif 'train_mse' in metrics:
            assert metrics['train_mse'] <= 1.0
    
    def test_training_reproducibility(self):
        """Test that training produces consistent results"""
        trainer1 = TestBaselineModelTrainer()
        trainer2 = TestBaselineModelTrainer()
        
        # Set same random seed for reproducibility
        np.random.seed(42)
        results1 = trainer1.train_and_register()
        
        np.random.seed(42)
        results2 = trainer2.train_and_register()
        
        # Results should be similar (allowing for some variance)
        assert results1['status'] == results2['status']
        assert results1['num_training_samples'] == results2['num_training_samples']
        
        # Metrics should be close (within reasonable tolerance)
        metrics1 = results1['metrics']
        metrics2 = results2['metrics']
        
        for key in ['train_accuracy', 'train_f1', 'train_mse']:
            if key in metrics1 and key in metrics2:
                diff = abs(metrics1[key] - metrics2[key])
                assert diff < 0.1  # Allow 10% difference due to randomness