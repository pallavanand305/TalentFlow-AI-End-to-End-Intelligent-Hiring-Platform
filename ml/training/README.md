# TalentFlow AI Model Training

This directory contains training scripts for TalentFlow AI machine learning models.

## Baseline TF-IDF Model Training

The baseline model uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization with cosine similarity to compute candidate-job matching scores.

### Files

- `train_baseline_model.py` - Main training script for production use
- `test_baseline_training.py` - Test version that works without database
- `README.md` - This documentation file

### Features

The baseline model training script:

1. **Loads training data** from the database (candidates, jobs, existing scores)
2. **Trains TF-IDF vectorizers** for three sections:
   - Skills (40% weight)
   - Experience (40% weight) 
   - Education (20% weight)
3. **Computes similarity scores** using cosine similarity
4. **Evaluates model performance** with train/validation split
5. **Logs metrics and parameters** to MLflow
6. **Registers the model** in MLflow model registry

### Usage

#### Production Training (requires database)

```bash
# Run with default settings
python ml/training/train_baseline_model.py

# Run with debug logging
python ml/training/train_baseline_model.py --log-level DEBUG

# Or use the convenience script
python scripts/train_baseline.py
```

#### Test Training (no database required)

```bash
# Run test version with synthetic data
python ml/training/test_baseline_training.py

# Run with debug logging
python ml/training/test_baseline_training.py --log-level DEBUG
```

### Requirements

#### For Production Training

- PostgreSQL database with candidate and job data
- MLflow tracking server running
- Required Python packages (see requirements.txt)

#### For Test Training

- Only requires Python packages (no database or MLflow)

### Model Parameters

The baseline model uses these hyperparameters:

```python
{
    'max_features': 5000,           # Maximum TF-IDF features
    'ngram_range': (1, 2),          # Unigrams and bigrams
    'stop_words': 'english',        # Remove English stop words
    'lowercase': True,              # Convert to lowercase
    'skills_weight': 0.4,           # Skills section weight
    'experience_weight': 0.4,       # Experience section weight
    'education_weight': 0.2,        # Education section weight
    'min_score_threshold': 0.1,     # Minimum score
    'max_score_threshold': 1.0      # Maximum score
}
```

### Training Metrics

The training script computes and logs these metrics:

#### Classification Metrics (binary threshold at 0.5)
- **Accuracy**: Fraction of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

#### Regression Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error

#### Model Info
- **Training samples**: Number of candidate-job pairs used for training
- **Validation samples**: Number of pairs used for validation
- **Feature counts**: Number of TF-IDF features per section

### Data Requirements

#### Training Data Format

The training script expects:

1. **Candidates table** with:
   - `id`: Unique candidate identifier
   - `name`: Candidate name
   - `email`: Contact email
   - `parsed_data`: JSON with structured resume data
   - `skills`: Array of skill strings

2. **Jobs table** with:
   - `id`: Unique job identifier
   - `title`: Job title
   - `description`: Job description text
   - `required_skills`: Array of required skills
   - `experience_level`: Required experience level

3. **Scores table** (optional, for ground truth):
   - `candidate_id`: Reference to candidate
   - `job_id`: Reference to job
   - `score`: Similarity score (0.0 to 1.0)

#### Parsed Resume Data Structure

The `parsed_data` JSON should contain:

```json
{
    "raw_text": "Full resume text...",
    "sections": {
        "experience": "Work experience section...",
        "education": "Education section...",
        "skills": "Skills section..."
    },
    "work_experience": [
        {
            "company": "Company Name",
            "title": "Job Title",
            "description": "Job description...",
            "start_date": "2020-01",
            "end_date": "2023-12",
            "confidence": 0.85
        }
    ],
    "education": [
        {
            "institution": "University Name",
            "degree": "Bachelor of Science",
            "field_of_study": "Computer Science",
            "description": "Degree details...",
            "confidence": 0.90
        }
    ],
    "skills": [
        {
            "skill": "Python",
            "confidence": 0.8
        }
    ]
}
```

### MLflow Integration

The training script integrates with MLflow for:

1. **Experiment Tracking**: Logs parameters, metrics, and artifacts
2. **Model Registry**: Registers trained models with versions
3. **Artifact Storage**: Stores TF-IDF vectorizers and model metadata

#### MLflow Configuration

Set these environment variables:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_ARTIFACT_ROOT="./mlruns"
```

Or update `backend/app/core/config.py`:

```python
MLFLOW_TRACKING_URI: str = "http://localhost:5000"
MLFLOW_ARTIFACT_ROOT: str = "./mlruns"
```

### Output

#### Successful Training

```
INFO: Starting TalentFlow AI Baseline Model Training
INFO: Environment: development
INFO: MLflow Tracking URI: http://localhost:5000
INFO: Loading training data from database...
INFO: Loaded 150 candidates
INFO: Loaded 25 jobs
INFO: Loaded 500 existing scores
INFO: Created 500 training samples
INFO: Training baseline TF-IDF model...
INFO: Training completed. Validation accuracy: 0.750
INFO: Logging model to MLflow...
INFO: Model logged to MLflow with run_id: abc123def456
INFO: Registering model in MLflow registry...
INFO: Model registered as version: 2
INFO: Training completed successfully!
INFO: Model: baseline_tfidf v2
INFO: Run ID: abc123def456
INFO: Training samples: 500
INFO: Validation Accuracy: 0.750
INFO: Validation F1: 0.720
INFO: Validation MSE: 0.125
```

#### Model Registration

The trained model is registered in MLflow with:

- **Model Name**: `baseline_tfidf`
- **Version**: Auto-incremented (1, 2, 3, ...)
- **Stage**: `None` (can be promoted to `Staging` or `Production`)
- **Artifacts**: TF-IDF vectorizers for each section
- **Metadata**: Training parameters and performance metrics

### Troubleshooting

#### Common Issues

1. **Database Connection Error**
   ```
   ERROR: [WinError 1225] The remote computer refused the network connection
   ```
   - Ensure PostgreSQL is running
   - Check database URL in configuration
   - Verify network connectivity

2. **MLflow Connection Error**
   ```
   ERROR: Connection refused to MLflow tracking server
   ```
   - Start MLflow server: `mlflow server --host 0.0.0.0 --port 5000`
   - Check MLFLOW_TRACKING_URI configuration

3. **No Training Data**
   ```
   ERROR: No training data available
   ```
   - Ensure candidates table has parsed_data
   - Verify jobs table has active jobs
   - Check database permissions

4. **Import Errors**
   ```
   ImportError: cannot import name 'get_async_session'
   ```
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify project structure

#### Debug Mode

Run with debug logging for detailed information:

```bash
python ml/training/train_baseline_model.py --log-level DEBUG
```

This will show:
- Detailed database queries
- Feature extraction process
- Vectorizer fitting progress
- MLflow logging details

### Testing

Run the unit tests to verify training functionality:

```bash
# Run all training tests
python -m pytest tests/unit/test_baseline_training.py -v

# Run specific test
python -m pytest tests/unit/test_baseline_training.py::TestBaselineModelTraining::test_train_model_with_valid_data -v

# Run with coverage
python -m pytest tests/unit/test_baseline_training.py --cov=ml.training --cov-report=html
```

## Semantic Similarity Model Training

The semantic model uses sentence transformers for advanced semantic understanding of candidate-job matching.

### Files

- `train_semantic_model.py` - Main training script for production use
- `test_semantic_training.py` - Test version that works without database
- `README.md` - This documentation file

### Features

The semantic model training script:

1. **Fine-tunes sentence transformers** on domain-specific data
2. **Uses contrastive learning** with candidate-job pairs
3. **Supports multiple base models** (all-MiniLM-L6-v2, etc.)
4. **Implements section-wise training** (skills, experience, education)
5. **Logs comprehensive metrics** to MLflow
6. **Registers fine-tuned models** in MLflow model registry
7. **Supports GPU acceleration** when available

### Usage

#### Production Training (requires database)

```bash
# Run with default settings
python ml/training/train_semantic_model.py

# Run with custom base model
python ml/training/train_semantic_model.py --base-model all-mpnet-base-v2

# Run with custom training parameters
python ml/training/train_semantic_model.py --epochs 5 --batch-size 32 --learning-rate 1e-5

# Run with debug logging
python ml/training/train_semantic_model.py --log-level DEBUG

# Or use the convenience script
python scripts/train_semantic.py
```

#### Test Training (no database required)

```bash
# Run test version with synthetic data
python ml/training/test_semantic_training.py

# Run with custom base model
python ml/training/test_semantic_training.py --base-model sentence-transformers/all-MiniLM-L12-v2

# Run with debug logging
python ml/training/test_semantic_training.py --log-level DEBUG
```

### Model Parameters

The semantic model uses these hyperparameters:

```python
{
    'base_model': 'all-MiniLM-L6-v2',    # Base sentence transformer
    'max_seq_length': 512,               # Maximum sequence length
    'batch_size': 16,                    # Training batch size
    'learning_rate': 2e-5,               # Learning rate
    'num_epochs': 3,                     # Number of training epochs
    'warmup_steps': 100,                 # Warmup steps
    'evaluation_steps': 500,             # Evaluation frequency
    'skills_weight': 0.4,                # Skills section weight
    'experience_weight': 0.4,            # Experience section weight
    'education_weight': 0.2,             # Education section weight
    'fine_tuning_enabled': True,         # Enable fine-tuning
    'contrastive_learning': True,        # Use contrastive learning
    'temperature': 0.07,                 # Temperature for contrastive loss
    'margin': 0.5                        # Margin for contrastive loss
}
```

### Training Metrics

The semantic training script computes and logs these metrics:

#### Correlation Metrics
- **Validation Correlation**: Pearson correlation between predicted and true scores
- **Spearman Correlation**: Rank correlation for ranking quality

#### Classification Metrics (binary threshold at 0.5)
- **Accuracy**: Fraction of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

#### Regression Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error

#### Model Info
- **Training samples**: Number of candidate-job pairs used for training
- **Training pairs**: Number of text pairs for contrastive learning
- **Embedding dimension**: Size of the embedding vectors
- **Device**: Training device (CPU/CUDA)

### Base Models

Supported sentence transformer models:

#### Recommended Models
- **all-MiniLM-L6-v2**: Fast, good performance (384 dimensions)
- **all-MiniLM-L12-v2**: Better performance, slower (384 dimensions)
- **all-mpnet-base-v2**: Best performance, slowest (768 dimensions)

#### Specialized Models
- **all-distilroberta-v1**: RoBERTa-based (768 dimensions)
- **paraphrase-MiniLM-L6-v2**: Optimized for paraphrase detection
- **multi-qa-MiniLM-L6-cos-v1**: Optimized for question-answering

### Fine-tuning Process

The semantic model training uses these techniques:

1. **Contrastive Learning**: Learns to distinguish similar vs dissimilar pairs
2. **Section-wise Training**: Separate training for skills, experience, education
3. **Weighted Loss**: Combines losses from different sections
4. **Cosine Similarity Loss**: Optimizes for cosine similarity matching
5. **Gradient Accumulation**: Handles large effective batch sizes
6. **Learning Rate Scheduling**: Warmup and decay for stable training

### Data Requirements

#### Training Data Format

Same as baseline model, but optimized for semantic understanding:

1. **Text Quality**: Clean, well-formatted text performs better
2. **Domain Relevance**: Domain-specific data improves performance
3. **Pair Diversity**: Diverse positive and negative pairs help generalization
4. **Label Quality**: Accurate similarity scores are crucial

#### Contrastive Pairs

The training creates contrastive pairs:

```python
# Positive pairs (high similarity)
("Python machine learning", "ML engineer with Python")  # Score: 0.8

# Negative pairs (low similarity)  
("Frontend JavaScript", "Database administrator")        # Score: 0.2

# Neutral pairs (medium similarity)
("Software engineer", "Technical lead")                  # Score: 0.5
```

### MLflow Integration

Enhanced MLflow integration for semantic models:

1. **Model Artifacts**: Saves complete sentence transformer model
2. **Metadata**: Stores base model, fine-tuning parameters
3. **Embeddings**: Logs sample embeddings for analysis
4. **Comparison**: Compares with baseline TF-IDF model
5. **Versioning**: Automatic version management

#### Model Registry

The fine-tuned model is registered with:

- **Model Name**: `semantic_similarity`
- **Version**: Auto-incremented (1, 2, 3, ...)
- **Stage**: `None` (can be promoted to `Staging` or `Production`)
- **Artifacts**: Complete sentence transformer model
- **Metadata**: Base model, training parameters, performance metrics

### GPU Support

The training script automatically detects and uses GPU when available:

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor GPU usage during training
nvidia-smi -l 1
```

#### GPU Requirements
- **CUDA**: Version 11.0 or higher
- **Memory**: At least 4GB VRAM for batch_size=16
- **Compute**: Compute capability 6.0 or higher

### Output

#### Successful Training

```
INFO: Starting TalentFlow AI Semantic Model Training
INFO: Environment: development
INFO: MLflow Tracking URI: http://localhost:5000
INFO: Base Model: all-MiniLM-L6-v2
INFO: Device: CUDA
INFO: Loading training data from database...
INFO: Loaded 150 candidates
INFO: Loaded 25 jobs
INFO: Created 500 training samples
INFO: Preparing training pairs for contrastive learning...
INFO: Created 1500 training pairs
INFO: Fine-tuning semantic similarity model...
INFO: Starting fine-tuning with 1200 training examples
Epoch 1/3: 100%|██████████| 75/75 [02:30<00:00,  2.00s/it]
Validation correlation: 0.742
INFO: Fine-tuning completed. Validation correlation: 0.742
INFO: Logging semantic model to MLflow...
INFO: Semantic model logged to MLflow with run_id: def789ghi012
INFO: Registering semantic model in MLflow registry...
INFO: Semantic model registered as version: 3
INFO: Training completed successfully!
INFO: Model: semantic_similarity v3
INFO: Base Model: all-MiniLM-L6-v2
INFO: Run ID: def789ghi012
INFO: Training samples: 500
INFO: Training pairs: 1500
INFO: Device: CUDA
INFO: Validation Correlation: 0.742
INFO: Validation Accuracy: 0.825
INFO: Validation F1: 0.810
INFO: Validation MSE: 0.089
```

### Troubleshooting

#### Common Issues

1. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Reduce batch_size: `--batch-size 8`
   - Use smaller model: `--base-model all-MiniLM-L6-v2`
   - Enable gradient checkpointing

2. **Slow Training**
   ```
   Training is very slow on CPU
   ```
   - Install CUDA-enabled PyTorch
   - Use smaller model for CPU training
   - Reduce sequence length: `max_seq_length=256`

3. **Import Errors**
   ```
   ImportError: No module named 'sentence_transformers'
   ```
   - Install: `pip install sentence-transformers`
   - Update requirements.txt

4. **Poor Performance**
   ```
   Validation correlation is very low
   ```
   - Increase training epochs: `--epochs 5`
   - Use larger model: `--base-model all-mpnet-base-v2`
   - Check data quality and labels

### Testing

Run the unit tests to verify semantic training functionality:

```bash
# Run all semantic training tests
python -m pytest tests/unit/test_semantic_training.py -v

# Run test training script
python ml/training/test_semantic_training.py

# Run with coverage
python -m pytest tests/unit/test_semantic_training.py --cov=ml.training --cov-report=html
```

### Next Steps

After training the semantic model:

1. **Compare Performance**: Compare with baseline TF-IDF model in MLflow
2. **A/B Test**: Run both models in parallel to compare results
3. **Promote Model**: Move best performing model to `Production`
4. **Deploy Model**: Update scoring service to use semantic model
5. **Monitor Performance**: Track embedding quality and similarity scores
6. **Iterate**: Fine-tune with more domain-specific data

### Advanced Training

For more sophisticated models, see:

- `train_ensemble_model.py` - Ensemble of multiple models
- `train_neural_model.py` - Deep learning approach
- `train_domain_specific.py` - Domain-specific fine-tuning