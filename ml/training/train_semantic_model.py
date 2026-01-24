#!/usr/bin/env python3
"""
Training script for semantic similarity model using sentence transformers

This script fine-tunes a sentence transformer model on domain-specific data,
logs metrics and parameters to MLflow, and registers the model in the registry.

Requirements: 5.1, 5.2
"""

import asyncio
import logging
import sys
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.pytorch
import torch
import pickle
import json

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

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class SemanticModelTrainer:
    """Trainer for semantic similarity model using sentence transformers"""
    
    def __init__(self, base_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize trainer
        
        Args:
            base_model: Base sentence transformer model to fine-tune
        """
        self.model_name = "semantic_similarity"
        self.model_version = "1.0.0"
        self.base_model = base_model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Section weights for scoring
        self.section_weights = {
            'skills': 0.4,
            'experience': 0.4,
            'education': 0.2
        }
        
        # Training parameters
        self.params = {
            'base_model': base_model,
            'max_seq_length': 512,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 3,
            'warmup_steps': 100,
            'evaluation_steps': 500,
            'save_steps': 1000,
            'skills_weight': self.section_weights['skills'],
            'experience_weight': self.section_weights['experience'],
            'education_weight': self.section_weights['education'],
            'device': str(self.device),
            'fine_tuning_enabled': True,
            'contrastive_learning': True,
            'temperature': 0.07,
            'margin': 0.5
        }
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading base model: {self.base_model}")
            self.model = SentenceTransformer(self.base_model, device=self.device)
            
            # Set max sequence length
            self.model.max_seq_length = self.params['max_seq_length']
            
            logger.info(f"Initialized semantic model on device: {self.device}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers library is required for semantic model training. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
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
            
            # Get existing scores for validation
            existing_scores = []
            for job in jobs[:20]:  # Limit to first 20 jobs
                job_scores = await score_repo.get_scores_for_job(job.id, current_only=True, limit=50)
                existing_scores.extend(job_scores)
            
            logger.info(f"Loaded {len(existing_scores)} existing scores")
            
            # Create training pairs
            training_data = []
            target_scores = []
            
            # If we have existing scores, use them as ground truth
            if existing_scores:
                logger.info("Using existing scores as ground truth")
                for score in existing_scores:
                    candidate = next((c for c in candidates if c.id == score.candidate_id), None)
                    job = next((j for j in jobs if j.id == score.job_id), None)
                    
                    if candidate and job and candidate.parsed_data:
                        feature_data = self._extract_features(candidate, job)
                        training_data.append(feature_data)
                        target_scores.append(float(score.score))
            
            # If no existing scores, generate synthetic training data
            if not training_data:
                logger.info("No existing scores found, generating synthetic training data")
                training_data, target_scores = await self._generate_synthetic_data(candidates, jobs)
            
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
                        exp_text = f"{exp.get('title', '')} at {exp.get('company', '')}. {exp.get('description', '')}"
                        experience_texts.append(exp_text)
            features['candidate_experience'] = ' '.join(experience_texts)
            
            # Education
            education_texts = []
            if 'education' in parsed_data:
                for edu in parsed_data['education']:
                    if isinstance(edu, dict):
                        edu_text = f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}. {edu.get('description', '')}"
                        education_texts.append(edu_text)
            features['candidate_education'] = ' '.join(education_texts)
        else:
            # Fallback to basic candidate data
            features['candidate_skills'] = ' '.join(candidate.skills or [])
            features['candidate_experience'] = ''
            features['candidate_education'] = candidate.education_level or ''
        
        # Extract job features
        features['job_skills'] = ' '.join(job.required_skills)
        features['job_experience'] = f"Looking for {job.experience_level.value} level experience. {job.description}"
        features['job_education'] = f"Job requirements: {job.description}"
        
        return features
    
    async def _generate_synthetic_data(self, candidates: List[Candidate], jobs: List[Job]) -> Tuple[List[Dict], List[float]]:
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
        
        # Generate pairs (limit to avoid too much data)
        max_pairs = min(1000, len(candidates) * min(10, len(jobs)))
        pairs_generated = 0
        
        for candidate in candidates:
            if pairs_generated >= max_pairs:
                break
                
            if not candidate.parsed_data:
                continue
            
            # Score against a few jobs using basic similarity
            for job in jobs[:min(5, len(jobs))]:
                if pairs_generated >= max_pairs:
                    break
                
                try:
                    feature_data = self._extract_features(candidate, job)
                    
                    # Generate synthetic score based on text similarity
                    synthetic_score = self._compute_synthetic_score(feature_data)
                    
                    training_data.append(feature_data)
                    target_scores.append(synthetic_score)
                    pairs_generated += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing candidate {candidate.id} for job {job.id}: {e}")
                    continue
        
        logger.info(f"Generated {len(training_data)} synthetic training samples")
        return training_data, target_scores
    
    def _compute_synthetic_score(self, features: Dict[str, str]) -> float:
        """
        Compute synthetic score for training data generation
        
        Args:
            features: Feature dictionary
            
        Returns:
            Synthetic similarity score
        """
        try:
            # Simple keyword-based similarity for synthetic data
            candidate_text = f"{features.get('candidate_skills', '')} {features.get('candidate_experience', '')} {features.get('candidate_education', '')}"
            job_text = f"{features.get('job_skills', '')} {features.get('job_experience', '')} {features.get('job_education', '')}"
            
            # Convert to lowercase and split into words
            candidate_words = set(candidate_text.lower().split())
            job_words = set(job_text.lower().split())
            
            # Compute Jaccard similarity
            if len(candidate_words) == 0 or len(job_words) == 0:
                return 0.0
            
            intersection = len(candidate_words.intersection(job_words))
            union = len(candidate_words.union(job_words))
            
            jaccard_sim = intersection / union if union > 0 else 0.0
            
            # Add some noise and normalize to [0, 1]
            noise = np.random.normal(0, 0.1)
            score = max(0.0, min(1.0, jaccard_sim + noise))
            
            return score
            
        except Exception as e:
            logger.warning(f"Error computing synthetic score: {e}")
            return 0.5  # Default score
    
    def prepare_training_pairs(self, training_data: List[Dict], target_scores: List[float]) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Prepare training pairs for contrastive learning
        
        Args:
            training_data: List of feature dictionaries
            target_scores: List of target scores
            
        Returns:
            Tuple of (text_pairs, similarity_scores)
        """
        logger.info("Preparing training pairs for contrastive learning...")
        
        text_pairs = []
        similarity_scores = []
        
        for features, score in zip(training_data, target_scores):
            # Create pairs for each section
            sections = ['skills', 'experience', 'education']
            
            for section in sections:
                candidate_text = features.get(f'candidate_{section}', '').strip()
                job_text = features.get(f'job_{section}', '').strip()
                
                if candidate_text and job_text:
                    text_pairs.append((candidate_text, job_text))
                    # Weight the score by section importance
                    weighted_score = score * self.section_weights.get(section, 1.0)
                    similarity_scores.append(weighted_score)
        
        logger.info(f"Created {len(text_pairs)} training pairs")
        return text_pairs, similarity_scores
    
    def fine_tune_model(self, text_pairs: List[Tuple[str, str]], similarity_scores: List[float]) -> Dict[str, Any]:
        """
        Fine-tune the sentence transformer model
        
        Args:
            text_pairs: List of text pairs for training
            similarity_scores: List of similarity scores
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Fine-tuning semantic similarity model...")
        
        if not text_pairs:
            raise ValueError("No training pairs available")
        
        try:
            from sentence_transformers import InputExample, losses
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
            from torch.utils.data import DataLoader
            
            # Prepare training examples
            train_examples = []
            for (text1, text2), score in zip(text_pairs, similarity_scores):
                train_examples.append(InputExample(texts=[text1, text2], label=float(score)))
            
            # Split into train/validation
            if len(train_examples) > 10:
                train_examples, val_examples = train_test_split(
                    train_examples, test_size=0.2, random_state=42
                )
            else:
                val_examples = train_examples[:min(5, len(train_examples))]
            
            # Create data loaders
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.params['batch_size'])
            
            # Define loss function
            train_loss = losses.CosineSimilarityLoss(self.model)
            
            # Create evaluator
            if val_examples:
                val_sentences1 = [example.texts[0] for example in val_examples]
                val_sentences2 = [example.texts[1] for example in val_examples]
                val_scores = [example.label for example in val_examples]
                
                evaluator = EmbeddingSimilarityEvaluator(
                    val_sentences1, val_sentences2, val_scores,
                    name='validation'
                )
            else:
                evaluator = None
            
            # Fine-tune the model
            logger.info(f"Starting fine-tuning with {len(train_examples)} training examples")
            
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=self.params['num_epochs'],
                evaluation_steps=self.params['evaluation_steps'],
                warmup_steps=self.params['warmup_steps'],
                output_path=None,  # Don't save during training
                save_best_model=False,
                show_progress_bar=True
            )
            
            # Evaluate the model
            metrics = self._evaluate_model(val_examples if val_examples else train_examples)
            
            logger.info(f"Fine-tuning completed. Validation correlation: {metrics.get('val_correlation', 'N/A')}")
            return metrics
            
        except ImportError as e:
            logger.error(f"Required libraries not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise
    
    def _evaluate_model(self, examples: List) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model
        
        Args:
            examples: List of evaluation examples
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating fine-tuned model...")
        
        try:
            # Extract texts and labels
            sentences1 = [example.texts[0] for example in examples]
            sentences2 = [example.texts[1] for example in examples]
            true_scores = [example.label for example in examples]
            
            # Generate embeddings
            embeddings1 = self.model.encode(sentences1, convert_to_tensor=False)
            embeddings2 = self.model.encode(sentences2, convert_to_tensor=False)
            
            # Compute predicted similarities
            predicted_scores = []
            for emb1, emb2 in zip(embeddings1, embeddings2):
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                predicted_scores.append(float(similarity))
            
            # Compute metrics
            # Correlation
            correlation = np.corrcoef(true_scores, predicted_scores)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Convert to binary classification for additional metrics
            threshold = 0.5
            true_binary = [1 if score >= threshold else 0 for score in true_scores]
            pred_binary = [1 if score >= threshold else 0 for score in predicted_scores]
            
            # Classification metrics
            accuracy = accuracy_score(true_binary, pred_binary)
            precision = precision_score(true_binary, pred_binary, zero_division=0)
            recall = recall_score(true_binary, pred_binary, zero_division=0)
            f1 = f1_score(true_binary, pred_binary, zero_division=0)
            
            # Regression metrics
            mse = np.mean([(p - t)**2 for p, t in zip(predicted_scores, true_scores)])
            mae = np.mean([abs(p - t) for p, t in zip(predicted_scores, true_scores)])
            
            metrics = {
                'val_correlation': correlation,
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1,
                'val_mse': mse,
                'val_mae': mae,
                'num_eval_samples': len(examples),
                'embedding_dimension': len(embeddings1[0]) if embeddings1 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {
                'val_correlation': 0.0,
                'val_accuracy': 0.0,
                'val_precision': 0.0,
                'val_recall': 0.0,
                'val_f1': 0.0,
                'val_mse': 1.0,
                'val_mae': 1.0,
                'num_eval_samples': len(examples),
                'embedding_dimension': 0
            }
    
    async def log_to_mlflow(self, metrics: Dict[str, Any]) -> str:
        """
        Log model, metrics, and parameters to MLflow
        
        Args:
            metrics: Training metrics
            
        Returns:
            MLflow run ID
        """
        logger.info("Logging semantic model to MLflow...")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(self.params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Create a temporary directory for model artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the sentence transformer model
                model_path = os.path.join(temp_dir, "sentence_transformer")
                self.model.save(model_path)
                
                # Log the model directory as an artifact
                mlflow.log_artifacts(model_path, "model")
                
                # Save additional metadata
                metadata = {
                    'model_type': 'sentence_transformer',
                    'base_model': self.base_model,
                    'section_weights': self.section_weights,
                    'device': str(self.device),
                    'max_seq_length': self.params['max_seq_length']
                }
                
                metadata_path = os.path.join(temp_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                mlflow.log_artifact(metadata_path, "model")
                
                # Log model using MLflow's pytorch integration
                try:
                    # Create a wrapper for the sentence transformer
                    class SentenceTransformerWrapper(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        
                        def forward(self, input_ids, attention_mask=None):
                            # This is a simplified forward pass
                            return self.model.encode(input_ids)
                    
                    wrapper = SentenceTransformerWrapper(self.model)
                    
                    # Log as PyTorch model
                    mlflow.pytorch.log_model(
                        wrapper,
                        "pytorch_model",
                        registered_model_name=self.model_name
                    )
                    
                except Exception as e:
                    logger.warning(f"Could not log PyTorch model: {e}")
                
                # Set tags
                mlflow.set_tag("model_type", "semantic_similarity")
                mlflow.set_tag("model_version", self.model_version)
                mlflow.set_tag("framework", "sentence_transformers")
                mlflow.set_tag("base_model", self.base_model)
                mlflow.set_tag("training_script", "train_semantic_model.py")
                mlflow.set_tag("fine_tuned", "true")
            
            run_id = run.info.run_id
            logger.info(f"Semantic model logged to MLflow with run_id: {run_id}")
            return run_id
    
    async def register_model(self, run_id: str) -> str:
        """
        Register model in MLflow model registry
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Model version
        """
        logger.info("Registering semantic model in MLflow registry...")
        
        try:
            # Register model using the model registry service
            model_version = await model_registry.register_model(
                run_id=run_id,
                model_name=self.model_name,
                description=f"Semantic similarity model v{self.model_version} fine-tuned on domain data using {self.base_model}"
            )
            
            logger.info(f"Semantic model registered as version: {model_version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register semantic model: {e}")
            raise
    
    async def train_and_register(self) -> Dict[str, Any]:
        """
        Complete training pipeline: load data, fine-tune, log to MLflow, and register
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting semantic model training pipeline...")
        
        try:
            # Load training data
            training_data, target_scores = await self.load_training_data()
            
            if not training_data:
                raise ValueError("No training data available")
            
            # Prepare training pairs
            text_pairs, similarity_scores = self.prepare_training_pairs(training_data, target_scores)
            
            if not text_pairs:
                raise ValueError("No training pairs could be created")
            
            # Fine-tune model
            metrics = self.fine_tune_model(text_pairs, similarity_scores)
            
            # Log to MLflow
            run_id = await self.log_to_mlflow(metrics)
            
            # Register model
            model_version = await self.register_model(run_id)
            
            # Prepare results
            results = {
                'model_name': self.model_name,
                'model_version': model_version,
                'run_id': run_id,
                'base_model': self.base_model,
                'metrics': metrics,
                'num_training_samples': len(training_data),
                'num_training_pairs': len(text_pairs),
                'device': str(self.device),
                'status': 'success'
            }
            
            logger.info("Semantic model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Semantic model training pipeline failed: {e}")
            return {
                'model_name': self.model_name,
                'base_model': self.base_model,
                'status': 'failed',
                'error': str(e)
            }


async def main():
    """Main training script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train semantic similarity model")
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base sentence transformer model (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting TalentFlow AI Semantic Model Training")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"MLflow Tracking URI: {settings.MLFLOW_TRACKING_URI}")
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create trainer and run training
    trainer = SemanticModelTrainer(base_model=args.base_model)
    
    # Update parameters from command line arguments
    trainer.params.update({
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    })
    
    try:
        results = await trainer.train_and_register()
        
        if results['status'] == 'success':
            logger.info("Training completed successfully!")
            logger.info(f"Model: {results['model_name']} v{results['model_version']}")
            logger.info(f"Base Model: {results['base_model']}")
            logger.info(f"Run ID: {results['run_id']}")
            logger.info(f"Training samples: {results['num_training_samples']}")
            logger.info(f"Training pairs: {results['num_training_pairs']}")
            logger.info(f"Device: {results['device']}")
            
            # Print key metrics
            metrics = results['metrics']
            logger.info(f"Validation Correlation: {metrics.get('val_correlation', 0):.3f}")
            logger.info(f"Validation Accuracy: {metrics.get('val_accuracy', 0):.3f}")
            logger.info(f"Validation F1: {metrics.get('val_f1', 0):.3f}")
            logger.info(f"Validation MSE: {metrics.get('val_mse', 0):.3f}")
            
        else:
            logger.error(f"Training failed: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())