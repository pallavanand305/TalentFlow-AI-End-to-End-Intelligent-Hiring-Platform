#!/usr/bin/env python3
"""
Test script for semantic model training (no database required)

This script tests the semantic model training functionality using synthetic data,
without requiring a database connection or MLflow server.
"""

import asyncio
import logging
import sys
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockCandidate:
    """Mock candidate for testing"""
    def __init__(self, candidate_id: str, skills: List[str], experience: str, education: str):
        self.id = candidate_id
        self.skills = skills
        self.parsed_data = {
            'skills': [{'skill': skill} for skill in skills],
            'work_experience': [
                {
                    'title': 'Software Engineer',
                    'company': 'Tech Corp',
                    'description': experience
                }
            ],
            'education': [
                {
                    'degree': 'Bachelor of Science',
                    'field_of_study': 'Computer Science',
                    'institution': 'University',
                    'description': education
                }
            ]
        }


class MockJob:
    """Mock job for testing"""
    def __init__(self, job_id: str, title: str, skills: List[str], description: str):
        self.id = job_id
        self.title = title
        self.required_skills = skills
        self.description = description
        
        # Mock experience level enum
        class ExperienceLevel:
            def __init__(self, value):
                self.value = value
        
        self.experience_level = ExperienceLevel('mid')


class TestSemanticModelTrainer:
    """Test version of semantic model trainer"""
    
    def __init__(self, base_model: str = "all-MiniLM-L6-v2"):
        """Initialize test trainer"""
        self.model_name = "test_semantic_similarity"
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
            'batch_size': 8,  # Smaller batch size for testing
            'learning_rate': 2e-5,
            'num_epochs': 1,  # Single epoch for testing
            'warmup_steps': 10,
            'evaluation_steps': 50,
            'save_steps': 100,
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
    
    def generate_test_data(self) -> Tuple[List[Dict], List[float]]:
        """Generate synthetic test data"""
        logger.info("Generating synthetic test data...")
        
        # Create mock candidates
        candidates = [
            MockCandidate(
                "cand1", 
                ["Python", "Machine Learning", "TensorFlow"],
                "Developed ML models for recommendation systems",
                "Computer Science degree with focus on AI"
            ),
            MockCandidate(
                "cand2",
                ["Java", "Spring Boot", "Microservices"],
                "Built scalable backend services",
                "Software Engineering degree"
            ),
            MockCandidate(
                "cand3",
                ["JavaScript", "React", "Node.js"],
                "Frontend development with modern frameworks",
                "Web Development bootcamp"
            ),
            MockCandidate(
                "cand4",
                ["Python", "Django", "PostgreSQL"],
                "Full-stack web development",
                "Computer Science degree"
            ),
            MockCandidate(
                "cand5",
                ["Data Science", "Python", "Pandas", "Scikit-learn"],
                "Data analysis and machine learning projects",
                "Statistics degree with data science specialization"
            )
        ]
        
        # Create mock jobs
        jobs = [
            MockJob(
                "job1",
                "ML Engineer",
                ["Python", "Machine Learning", "TensorFlow", "PyTorch"],
                "Looking for ML engineer to build recommendation systems"
            ),
            MockJob(
                "job2",
                "Backend Developer",
                ["Java", "Spring Boot", "Microservices", "Docker"],
                "Backend developer for scalable microservices architecture"
            ),
            MockJob(
                "job3",
                "Frontend Developer",
                ["JavaScript", "React", "TypeScript", "CSS"],
                "Frontend developer for modern web applications"
            ),
            MockJob(
                "job4",
                "Full Stack Developer",
                ["Python", "Django", "React", "PostgreSQL"],
                "Full-stack developer for web applications"
            ),
            MockJob(
                "job5",
                "Data Scientist",
                ["Python", "Data Science", "Machine Learning", "SQL"],
                "Data scientist for analytics and ML projects"
            )
        ]
        
        # Generate training pairs
        training_data = []
        target_scores = []
        
        for candidate in candidates:
            for job in jobs:
                features = self._extract_features(candidate, job)
                score = self._compute_synthetic_score(features)
                
                training_data.append(features)
                target_scores.append(score)
        
        logger.info(f"Generated {len(training_data)} training samples")
        return training_data, target_scores
    
    def _extract_features(self, candidate: MockCandidate, job: MockJob) -> Dict[str, str]:
        """Extract features from candidate and job"""
        features = {}
        
        # Extract candidate features
        parsed_data = candidate.parsed_data
        
        # Skills
        skills = parsed_data['skills']
        features['candidate_skills'] = ' '.join([skill['skill'] for skill in skills])
        
        # Experience
        experience_texts = []
        for exp in parsed_data['work_experience']:
            exp_text = f"{exp['title']} at {exp['company']}. {exp['description']}"
            experience_texts.append(exp_text)
        features['candidate_experience'] = ' '.join(experience_texts)
        
        # Education
        education_texts = []
        for edu in parsed_data['education']:
            edu_text = f"{edu['degree']} in {edu['field_of_study']} from {edu['institution']}. {edu['description']}"
            education_texts.append(edu_text)
        features['candidate_education'] = ' '.join(education_texts)
        
        # Extract job features
        features['job_skills'] = ' '.join(job.required_skills)
        features['job_experience'] = f"Looking for {job.experience_level.value} level experience. {job.description}"
        features['job_education'] = f"Job requirements: {job.description}"
        
        return features
    
    def _compute_synthetic_score(self, features: Dict[str, str]) -> float:
        """Compute synthetic score for training data generation"""
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
            noise = np.random.normal(0, 0.05)  # Less noise for testing
            score = max(0.0, min(1.0, jaccard_sim + noise))
            
            return score
            
        except Exception as e:
            logger.warning(f"Error computing synthetic score: {e}")
            return 0.5  # Default score
    
    def prepare_training_pairs(self, training_data: List[Dict], target_scores: List[float]) -> Tuple[List[Tuple[str, str]], List[float]]:
        """Prepare training pairs for contrastive learning"""
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
        """Fine-tune the sentence transformer model"""
        logger.info("Fine-tuning semantic similarity model...")
        
        if not text_pairs:
            raise ValueError("No training pairs available")
        
        try:
            from sentence_transformers import InputExample, losses
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
            from torch.utils.data import DataLoader
            from sklearn.model_selection import train_test_split
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Prepare training examples
            train_examples = []
            for (text1, text2), score in zip(text_pairs, similarity_scores):
                train_examples.append(InputExample(texts=[text1, text2], label=float(score)))
            
            # Split into train/validation (use smaller validation set for testing)
            if len(train_examples) > 4:
                train_examples, val_examples = train_test_split(
                    train_examples, test_size=0.3, random_state=42
                )
            else:
                val_examples = train_examples[:2]  # Use first 2 for validation
            
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
        """Evaluate the fine-tuned model"""
        logger.info("Evaluating fine-tuned model...")
        
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics.pairwise import cosine_similarity
            
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
    
    def test_model_inference(self) -> Dict[str, Any]:
        """Test model inference capabilities"""
        logger.info("Testing model inference...")
        
        try:
            # Test sentences
            test_sentences = [
                "Python machine learning engineer with TensorFlow experience",
                "Java backend developer with Spring Boot",
                "Frontend developer with React and JavaScript",
                "Data scientist with Python and pandas"
            ]
            
            # Generate embeddings
            embeddings = self.model.encode(test_sentences, convert_to_tensor=False)
            
            # Compute pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            # Test results
            results = {
                'embedding_dimension': len(embeddings[0]),
                'num_test_sentences': len(test_sentences),
                'avg_similarity': float(np.mean(similarity_matrix)),
                'max_similarity': float(np.max(similarity_matrix)),
                'min_similarity': float(np.min(similarity_matrix)),
                'inference_successful': True
            }
            
            logger.info(f"Inference test completed. Embedding dimension: {results['embedding_dimension']}")
            return results
            
        except Exception as e:
            logger.error(f"Error during inference test: {e}")
            return {
                'embedding_dimension': 0,
                'num_test_sentences': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'inference_successful': False,
                'error': str(e)
            }
    
    def run_test_training(self) -> Dict[str, Any]:
        """Run complete test training pipeline"""
        logger.info("Starting test semantic model training pipeline...")
        
        try:
            # Generate test data
            training_data, target_scores = self.generate_test_data()
            
            if not training_data:
                raise ValueError("No training data available")
            
            # Prepare training pairs
            text_pairs, similarity_scores = self.prepare_training_pairs(training_data, target_scores)
            
            if not text_pairs:
                raise ValueError("No training pairs could be created")
            
            # Fine-tune model
            metrics = self.fine_tune_model(text_pairs, similarity_scores)
            
            # Test inference
            inference_results = self.test_model_inference()
            
            # Combine results
            results = {
                'model_name': self.model_name,
                'base_model': self.base_model,
                'training_metrics': metrics,
                'inference_results': inference_results,
                'num_training_samples': len(training_data),
                'num_training_pairs': len(text_pairs),
                'device': str(self.device),
                'status': 'success'
            }
            
            logger.info("Test semantic model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Test training pipeline failed: {e}")
            return {
                'model_name': self.model_name,
                'base_model': self.base_model,
                'status': 'failed',
                'error': str(e)
            }


async def main():
    """Main test script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test semantic similarity model training")
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base sentence transformer model (default: all-MiniLM-L6-v2)"
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
    
    logger.info("Starting TalentFlow AI Semantic Model Training Test")
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create test trainer and run training
    trainer = TestSemanticModelTrainer(base_model=args.base_model)
    
    try:
        results = await asyncio.get_event_loop().run_in_executor(
            None, trainer.run_test_training
        )
        
        if results['status'] == 'success':
            logger.info("Test training completed successfully!")
            logger.info(f"Model: {results['model_name']}")
            logger.info(f"Base Model: {results['base_model']}")
            logger.info(f"Training samples: {results['num_training_samples']}")
            logger.info(f"Training pairs: {results['num_training_pairs']}")
            logger.info(f"Device: {results['device']}")
            
            # Print key metrics
            metrics = results['training_metrics']
            logger.info(f"Validation Correlation: {metrics.get('val_correlation', 0):.3f}")
            logger.info(f"Validation Accuracy: {metrics.get('val_accuracy', 0):.3f}")
            logger.info(f"Validation F1: {metrics.get('val_f1', 0):.3f}")
            
            # Print inference results
            inference = results['inference_results']
            if inference['inference_successful']:
                logger.info(f"Embedding Dimension: {inference['embedding_dimension']}")
                logger.info(f"Average Similarity: {inference['avg_similarity']:.3f}")
            else:
                logger.warning(f"Inference test failed: {inference.get('error', 'Unknown error')}")
            
        else:
            logger.error(f"Test training failed: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test training script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())