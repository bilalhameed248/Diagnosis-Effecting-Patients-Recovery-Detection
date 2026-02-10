# Clinical Text Classification with Bio_ClinicalBERT

A deep learning based project for automated classification of clinical assessment text to identify whether "Diagnosis affecting recovery" is present in initial evaluations.

## 📋 Overview

This project implements a fine-tuned BERT-based model (Bio_ClinicalBERT) to analyze clinical assessment text and determine if the documentation contains evidence of diagnosis affecting patient recovery. The model processes clinical impressions and evaluations to support healthcare documentation quality assurance by identifying whether recovery-impacting diagnoses are properly documented.

## 🎯 Key Features

- **Pre-trained Clinical Model**: Utilizes `emilyalsentzer/Bio_ClinicalBERT` optimized for clinical text
- **Text Preprocessing**: Advanced NLP preprocessing with lemmatization and stopword removal
- **Binary Classification**: Identifies presence/absence of diagnosis affecting recovery
- **High Accuracy Threshold**: 95% confidence threshold for predictions
- **Sentence-Level Analysis**: Tokenizes and analyzes text at sentence granularity
- **GPU Acceleration**: CUDA-enabled training and inference

## 📁 Project Structure

```
.
├── fd_train.csv           # Training dataset
├── fd_test.csv            # Testing dataset
├── requirements.txt       # Python dependencies
├── Untitled.ipynb         # Main notebook with training & inference
├── save_tokenizer_bcb/    # Saved tokenizer (generated)
├── save_model_bcb/        # Saved base model (generated)
└── fine_tuned_model_bcb/  # Fine-tuned model checkpoints (generated)
```

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **NLP**: NLTK, WordNetLemmatizer
- **Data Processing**: Pandas, NumPy
- **Model Training**: Hugging Face Trainer API
- **Evaluation**: scikit-learn, Hugging Face Evaluate
- **Visualization**: Seaborn
- **GPU Management**: CUDA, Numba, pynvml

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bilalhameed248/Diagnosis-Effecting-Patients-Recovery-Detection.git
   cd Diagnosis-Effecting-Patients-Recovery-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('popular')
   ```

## 🚀 Usage

### Training the Model

1. **Prepare your data**: Ensure `fd_train.csv` and `fd_test.csv` contain columns:
   - `sentence`: Clinical text
   - `label`: Binary labels (0/1)

2. **Run training cells** in [Untitled.ipynb](Untitled.ipynb):
   ```python
   # Load and preprocess data
   dataset = load_dataset('csv', data_files={
       'train': './fd_train.csv', 
       'test': './fd_test.csv'
   })
   
   # Train model
   trainer.train()
   ```

3. **Training configuration**:
   - Epochs: 10
   - Batch size: 4
   - Max sequence length: 512 tokens
   - Evaluation strategy: Per epoch

### Inference

Use the `inference_why_skill_care()` function for predictions:

```python
content = """
Patient evaluation text here...
"""

inference_why_skill_care(content)
```

**Output**: Returns whether diagnosis affecting recovery is found with 95%+ confidence.

## 📊 Model Performance

The model includes:
- **Accuracy metric** evaluation
- **Confusion matrix** analysis
- **Classification reports**
- Checkpoint saving (top 2 models)

## 🔧 Configuration

### Training Arguments
```python
TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    save_total_limit=2,
    evaluation_strategy="epoch"
)
```

### Text Preprocessing
- Lemmatization using WordNet
- Stopword removal (English)
- Lowercasing and special character removal
- Whitespace normalization

## 💻 Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended
- **VRAM**: Minimum 8GB for batch size 4
- **Storage**: ~2GB for model checkpoints

Monitor GPU usage:
```python
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
```

## 📈 Results

The model evaluates clinical documentation to determine if it contains evidence of diagnosis affecting patient recovery. Key metrics tracked:
- Training/validation accuracy
- Confusion matrix
- Per-epoch performance

## 📧 Contact

For questions or support, please open an issue in the repository.

---
````
