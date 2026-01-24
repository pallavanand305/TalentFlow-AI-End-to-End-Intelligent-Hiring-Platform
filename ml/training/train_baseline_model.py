#!/usr/bin/env python3
"""
Training script for baseline TF-IDF model

This script loads training data from the database, trains a TF-IDF model,
logs metrics and parameters to MLflow, and registers the model in the registry.

Requirements: 5.1, 5.2
"""

import asyncio
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
from backend.app.core.database import AsyncSessionLocal
from backend.app.core.logging import setup_logging
from backend.app.repositories.candidate_repository import CandidateRepository
from backend.app.repositories.job_repository import JobRepository
from backend.app.repositories.score_repository import ScoreRepository
from backend.app.services.model_registry import model_registry
from backend.app.models.candidate import Candidate
from backend.app.models.job import Job
from backend.app.models.score import Score
from ml.inference.scoring_engine import ScoringEngine

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class BaselineModelTrainer:
    """Trainer for baseline TF-IDF model"""
    
    def __init__(self):
        """Initialize trainer"""
        self.model_name = "baseline_tfidf"
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
    
    async def load_training_data(self) -> Tuple[List[Dict], List[float]]:
        """
        Load training data from database
        
        Returns:
            Tuple of (feature_data, target_scores)
        """
        logger.info("Loading training data from database...")
        
        async with AsyncSessionLocal() as session:
            candidate_repo = CandidateRepository(session)
            job_repo = JobRepository(session)
            score_repo = ScoreRepository(session)
            
            # Get all candidates with parsed data
            candidates = await candidate_repo.list_all(limit=1000)
            logger.info(f"Loaded {len(candidates)} candidates")
            
            # Get all active jobs
            jobs = await job_repo.get_all(limit=1000)
            logger.info(f"Loaded {len(jobs)} jobs")
            
            # Get existing scores for validation (limit to recent scores)
            # Use a query to get scores from multiple jobs
            existing_scores = []
            for job in jobs[:10]:  # Limit to first 10 jobs to avoid too much data
                job_scores = await score_repo.get_scores_for_job(job.id, current_only=True, limit=100)
                existing_scores.extend(job_scores)
            
            logger.info(f"Loaded {len(existing_scores)} existing scores")
            
            # Create training pairs
            training_data = []
            target_scores = []
            
            # If we have existing scores, use them as ground truth
            if existing_scores:
                logger.info("Using existing scores as ground truth")
                for score in existing_scores:
                    # Find corresponding candidate and job
                    candidate = next((c for c in candidates if c.id == score.candidate_id), None)
                    job = next((j for j in jobs if j.id == score.job_id), None)
                    
                    if candidate and job and candidate.parsed_data:
                        feature_data = self._extract_features(candidate, job)
                        training_data.append(feature_data)
                        target_scores.append(float(score.score))
            
            # If no existing scores, generate synthetic training data
            if not training_data:
                logger.info("No existing scores found, generating synthetic training data")
                training_data, target_scores = self._generate_synthetic_data(candidates, jobs)
            
            logger.info(f"Created {len(training_data)} training samples")
            return training_data, target_scores
    
    def _extract_features(self, candidate: Candidate, job: Job) -> Dict[str, str]:
        """
        Extract features from candidate and job for training
        
        Args:
            candidate: Candidate object
            job: Job object
            
        Returns:
            Dictionary with feature texts
        """
        features = {}
        
        # Extract candidate features
        if candidate.parsed_data:
            parsed_data = candidate.parsed_data
            
            # Skills
            if 'skills' in parsed_data:
                skills = parsed_data['skills']
                if isinstance(skills, list):
                    features['candidate_skills'] = ' '.join([
                        skill.get('skill', '') if isinstance(skill, dict) else str(skill)
                        for skill in skills
                    ])
                else:
                    features['candidate_skills'] = str(skills)
            else:
                features['candidate_skills'] = ' '.join(candidate.skills or [])
            
            # Experience
            experience_texts = []
            if 'work_experience' in parsed_data:
                for exp in parsed_data['work_experience']:
                    if isinstance(exp, dict):
                        exp_text = f"{exp.get('title', '')} {exp.get('company', '')} {exp.get('description', '')}"
                        experience_texts.append(exp_text)
            features['candidate_experience'] = ' '.join(experience_texts)
            
            # Education
            education_texts = []
            if 'education' in parsed_data:
                for edu in parsed_data['education']:
                    if isinstance(edu, dict):
                        edu_text = f"{edu.get('degree', '')} {edu.get('field_of_study', '')} {edu.get('institution', '')}"
                        education_texts.append(edu_text)
            features['candidate_education'] = ' '.join(education_texts)
        else:
            # Fallback to basic candidate data
            features['candidate_skills'] = ' '.join(candidate.skills or [])
            features['candidate_experience'] = ''
            features['candidate_education'] = candidate.education_level or ''
        
        # Extract job features
        features['job_skills'] = ' '.join(job.required_skills)
        features['job_experience'] = f"{job.experience_level.value} {job.description}"
        features['job_education'] = job.description  # Extract education requirements from description
        
        return features
    
    def _generate_synthetic_data(self, candidates: List[Candidate], jobs: List[Job]) -> Tuple[List[Dict], List[float]]:
        """
        Generate synthetic training data when no existing scores are available
        
        Args:
            candidates: List of candidates
            jobs: List of jobs
            
        Returns:
            Tuple of (training_data, target_scores)
        """
        logger.info("Generating synthetic training data...")
        
        training_data = []
        target_scores = []
        
        # Create a temporary scoring engine to generate synthetic scores
        scoring_engine = ScoringEngine(model_type="tfidf")
        
        # Generate pairs (limit to avoid too much data)
        max_pairs = min(1000, len(candidates) * min(10, len(jobs)))
        pairs_generated = 0
        
        for candidate in candidates:
            if pairs_generated >= max_pairs:
                break
                
            if not candidate.parsed_data:
                continue
                
            # Convert candidate data to ParsedResume format for scoring
            try:
                from backend.app.schemas.resume import ParsedResume, WorkExperience, Education, Skill, Certification
                
                # Extract work experience
                work_experience = []
                if 'work_experience' in candidate.parsed_data:
                    for exp in candidate.parsed_data['work_experience']:
                        if isinstance(exp, dict):
                            work_experience.append(WorkExperience(
                                company=exp.get('company', ''),
                                title=exp.get('title', ''),
                                start_date=exp.get('start_date'),
                                end_date=exp.get('end_date'),
                                description=exp.get('description', ''),
                                confidence=exp.get('confidence', 0.8)
                            ))
                
                # Extract education
                education = []
                if 'education' in candidate.parsed_data:
                    for edu in candidate.parsed_data['education']:
                        if isinstance(edu, dict):
                            education.append(Education(
                                institution=edu.get('institution', ''),
                                degree=edu.get('degree', ''),
                                field_of_study=edu.get('field_of_study'),
                                start_date=edu.get('start_date'),
                                end_date=edu.get('end_date'),
                                description=edu.get('description', ''),
                                confidence=edu.get('confidence', 0.8)
                            ))
                
                # Extract skills
                skills = []
                if 'skills' in candidate.parsed_data:
                    for skill in candidate.parsed_data['skills']:
                        if isinstance(skill, dict):
                            skills.append(Skill(
                                skill=skill.get('skill', ''),
                                confidence=skill.get('confidence', 0.8)
                            ))
                        else:
                            skills.append(Skill(skill=str(skill), confidence=0.8))
                
                # Extract certifications
                certifications = []
                if 'certifications' in candidate.parsed_data:
                    for cert in candidate.parsed_data['certifications']:
                        if isinstance(cert, dict):
                            certifications.append(Certification(
                                certification=cert.get('certification', ''),
                                date=cert.get('date'),
                                confidence=cert.get('confidence', 0.8)
                            ))
                        else:
                            certifications.append(Certification(certification=str(cert), confidence=0.8))
                
                parsed_resume = ParsedResume(
                    raw_text=candidate.parsed_data.get('raw_text', ''),
                    sections=candidate.parsed_data.get('sections', {}),
                    work_experience=work_experience,
                    education=education,
                    skills=skills,
                    certifications=certifications,
                    low_confidence_fields=candidate.parsed_data.get('low_confidence_fields', []),
                    file_format=candidate.parsed_data.get('file_format', 'unknown')
                )
                
                # Score against a few jobs
                for job in jobs[:min(5, len(jobs))]:
                    if pairs_generated >= max_pairs:
                        break
                    
                    try:
                        score_result = scoring_engine.score_candidate(parsed_resume, job)
                        
                        feature_data = self._extract_features(candidate, job)
                        training_data.append(feature_data)
                        target_scores.append(score_result['overall_score'])
                        pairs_generated += 1
                        
                    except Exception as e:
                        logger.warning(f"Error scoring candidate {candidate.id} for job {job.id}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error processing candidate {candidate.id}: {e}")
                continue
        
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
                # Get individual feature vectors
                skills_vec = skills_vectors[i:i+1]
                exp_vec = experience_vectors[i:i+1]
                edu_vec = education_vectors[i:i+1]
                
                # Compute section similarities (simplified for training)
                skills_sim = cosine_similarity(skills_vec, skills_vec)[0][0] if skills_vec.nnz > 0 else 0.0
                exp_sim = cosine_similarity(exp_vec, exp_vec)[0][0] if exp_vec.nnz > 0 else 0.0
                edu_sim = cosine_similarity(edu_vec, edu_vec)[0][0] if edu_vec.nnz > 0 else 0.0
                
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
        if len(training_data) > 10:
            X_train, X_val, y_train, y_val = train_test_split(
                list(range(len(training_data))), target_scores, test_size=0.2, random_state=42
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
    
    async def log_to_mlflow(self, metrics: Dict[str, Any]) -> str:
        """
        Log model, metrics, and parameters to MLflow
        
        Args:
            metrics: Training metrics
            
        Returns:
            MLflow run ID
        """
        logger.info("Logging model to MLflow...")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(self.params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Create a temporary directory for model artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save vectorizers
                skills_vectorizer_path = os.path.join(temp_dir, "skills_vectorizer.pkl")
                experience_vectorizer_path = os.path.join(temp_dir, "experience_vectorizer.pkl")
                education_vectorizer_path = os.path.join(temp_dir, "education_vectorizer.pkl")
                
                joblib.dump(self.skills_vectorizer, skills_vectorizer_path)
                joblib.dump(self.experience_vectorizer, experience_vectorizer_path)
                joblib.dump(self.education_vectorizer, education_vectorizer_path)
                
                # Log vectorizers as artifacts
                mlflow.log_artifact(skills_vectorizer_path, "model")
                mlflow.log_artifact(experience_vectorizer_path, "model")
                mlflow.log_artifact(education_vectorizer_path, "model")
                
                # Create a simple model wrapper for sklearn logging
                class BaselineTFIDFModel:
                    def __init__(self, skills_vec, exp_vec, edu_vec, weights):
                        self.skills_vectorizer = skills_vec
                        self.experience_vectorizer = exp_vec
                        self.education_vectorizer = edu_vec
                        self.section_weights = weights
                    
                    def predict(self, X):
                        # Placeholder predict method
                        return [0.5] * len(X)
                
                # Create model instance
                model = BaselineTFIDFModel(
                    self.skills_vectorizer,
                    self.experience_vectorizer,
                    self.education_vectorizer,
                    self.section_weights
                )
                
                # Log the model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=self.model_name
                )
                
                # Log additional metadata
                mlflow.set_tag("model_type", "baseline_tfidf")
                mlflow.set_tag("model_version", self.model_version)
                mlflow.set_tag("framework", "sklearn")
                mlflow.set_tag("training_script", "train_baseline_model.py")
            
            run_id = run.info.run_id
            logger.info(f"Model logged to MLflow with run_id: {run_id}")
            return run_id
    
    async def register_model(self, run_id: str) -> str:
        """
        Register model in MLflow model registry
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Model version
        """
        logger.info("Registering model in MLflow registry...")
        
        try:
            # Register model using the model registry service
            model_version = await model_registry.register_model(
                run_id=run_id,
                model_name=self.model_name,
                description=f"Baseline TF-IDF model v{self.model_version} for candidate-job matching"
            )
            
            logger.info(f"Model registered as version: {model_version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    async def train_and_register(self) -> Dict[str, Any]:
        """
        Complete training pipeline: load data, train, log to MLflow, and register
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting baseline model training pipeline...")
        
        try:
            # Load training data
            training_data, target_scores = await self.load_training_data()
            
            if not training_data:
                raise ValueError("No training data available")
            
            # Train model
            metrics = self.train_model(training_data, target_scores)
            
            # Log to MLflow
            run_id = await self.log_to_mlflow(metrics)
            
            # Register model
            model_version = await self.register_model(run_id)
            
            # Prepare results
            results = {
                'model_name': self.model_name,
                'model_version': model_version,
                'run_id': run_id,
                'metrics': metrics,
                'num_training_samples': len(training_data),
                'status': 'success'
            }
            
            logger.info("Baseline model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {
                'model_name': self.model_name,
                'status': 'failed',
                'error': str(e)
            }


async def main():
    """Main training script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF model")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting TalentFlow AI Baseline Model Training")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"MLflow Tracking URI: {settings.MLFLOW_TRACKING_URI}")
    
    # Create trainer and run training
    trainer = BaselineModelTrainer()
    
    try:
        results = await trainer.train_and_register()
        
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
            else:
                logger.info(f"Training Accuracy: {metrics['train_accuracy']:.3f}")
                logger.info(f"Training F1: {metrics['train_f1']:.3f}")
        else:
            logger.error(f"Training failed: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())