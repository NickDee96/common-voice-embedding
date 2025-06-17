# Building Linguistically Robust RAG Systems with Common Voice Transcriptions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/thinkKenya)

> **Leveraging Mozilla Common Voice transcriptions to enhance text-based Retrieval-Augmented Generation (RAG) systems for better query diversity handling and coverage of underrepresented languages.**

## 🎯 Overview

This project demonstrates a novel approach to enhancing RAG systems using **Mozilla Common Voice transcriptions**. Instead of traditional text corpora, we leverage speech-derived data to bridge the "formality gap" between how users actually speak and how knowledge bases are structured.

### 🏆 Key Achievements

- **92.4% Accuracy@1** (up from 72.2% baseline)
- **95.9% Accuracy@5** (up from 82.8% baseline) 
- **0.944 NDCG@10** (up from 0.791 baseline)
- **0.938 MRR@10** (up from 0.767 baseline)

### 🌟 Why This Matters

Traditional RAG systems struggle with:
- **The "Formality Gap"**: Users speak colloquially, but knowledge bases use formal text
- **Linguistic Diversity**: Underrepresented languages lack sufficient coverage
- **Query Variations**: Paraphrased and vernacular expressions are poorly handled

Our solution uses **Common Voice transcriptions** to capture natural language variations that traditional text corpora miss.

## 🚀 Quick Start

### Prerequisites

```bash
pip install datasets llama-index-llms-openai llama-index-embeddings-openai
pip install llama-index-finetuning llama-index-embeddings-huggingface
pip install "transformers[torch]" sentence-transformers
pip install python-dotenv tqdm pandas scikit-learn
```

### Environment Setup

Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Run the Notebook

```bash
jupyter notebook finetune_embedding.ipynb
```

## 📊 Results Summary

| Metric | BGE Baseline | Fine-tuned | Improvement |
|--------|-------------|------------|-------------|
| **Accuracy@1** | 72.2% | 92.4% | **+28.0%** |
| **Accuracy@5** | 82.8% | 95.9% | **+15.8%** |
| **MRR@10** | 0.767 | 0.938 | **+22.3%** |
| **NDCG@10** | 0.791 | 0.944 | **+19.3%** |
| **Recall@10** | 86.7% | 96.3% | **+11.1%** |

## 🔬 Technical Approach

### 1. Data Preparation
```python
from datasets import load_dataset

# Load Common Voice Swahili transcriptions
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "sw")
transcriptions = dataset["train"]["sentence"]
```

### 2. Synthetic Q&A Generation
```python
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.llms.openai import OpenAI

# Generate diverse Q&A pairs from transcriptions
qa_dataset = generate_qa_embedding_pairs(
    llm=OpenAI(model="gpt-4o"),
    nodes=text_nodes
)
```

### 3. Embedding Fine-tuning
```python
from llama_index.finetuning import SentenceTransformersFinetuneEngine

# Fine-tune BAAI/bge-small-en model
finetune_engine = SentenceTransformersFinetuneEngine(
    qa_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="enhanced_model"
)
finetune_engine.finetune()
```

## 🏗️ Project Structure

```
common-voice-embedding/
├── finetune_embedding.ipynb    # Main training notebook
├── presentation.md             # Technical presentation
├── abstract.md                # Research abstract
├── common_voice_swahili_*.csv  # Sampled datasets
├── train_dataset.json         # Generated Q&A pairs
├── val_dataset.json          # Validation Q&A pairs
├── test_model/               # Fine-tuned model artifacts
│   ├── README.md            # Model card
│   ├── config.json          # Model configuration
│   └── model.safetensors    # Model weights
└── results/                 # Evaluation results
    ├── Information-Retrieval_evaluation_bge_results.csv
    └── Information-Retrieval_evaluation_finetuned_results.csv
```

## 🔧 Model Architecture

Our fine-tuned model is based on **BAAI/bge-small-en** with the following specifications:

```python
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True}) 
      with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 
                'pooling_mode_cls_token': True})
  (2): Normalize()
)
```

**Key Specifications:**
- **Output Dimensionality:** 384 dimensions
- **Maximum Sequence Length:** 512 tokens
- **Similarity Function:** Cosine similarity
- **Loss Function:** MultipleNegativesRankingLoss
- **Training Epochs:** 2 (highly efficient!)

## 📈 Training Progression

| Epoch | Step | NDCG@10 |
|:-----:|:----:|:-------:|
| 0.17  | 50   | 0.8917  |
| 0.33  | 100  | 0.9226  |
| 0.50  | 150  | 0.9298  |
| 1.00  | 300  | 0.9346  |
| **2.00** | **600** | **0.9443** |

## 🌍 Impact for Underrepresented Languages

### Why Swahili?
- **150+ million speakers** across East Africa
- **Well-represented** in Common Voice with active community
- **Underrepresented** in traditional NLP datasets
- **Natural variations** between formal and spoken Swahili

### Broader Applications
- Extend to other African languages (Yoruba, Hausa, Amharic)
- Apply to global underrepresented languages
- Enhance multilingual customer support
- Improve educational AI assistants

## 🔍 Evaluation Metrics Explained

### Accuracy@K
**What it measures:** Percentage of queries where the correct document appears in top-K results
- **Accuracy@1 = 92.4%:** 92 out of 100 queries get perfect answers as #1 result
- **Accuracy@5 = 95.9%:** 96 out of 100 queries succeed in top-5 results

### Mean Reciprocal Rank (MRR)
**What it measures:** Average of reciprocal ranks (1/position) of first correct result
- **MRR@10 = 0.938:** Most answers appear very high in rankings
- **Real impact:** Users find answers with minimal scrolling

### NDCG@10
**What it measures:** Ranking quality considering position-based relevance decay
- **NDCG@10 = 0.944:** Near-perfect ranking quality (1.0 is theoretical max)
- **Real impact:** Most relevant content consistently prioritized

## 🛠️ Usage Examples

### Loading the Fine-tuned Model

```python
from sentence_transformers import SentenceTransformer

# Load our fine-tuned model
model = SentenceTransformer("./test_model")

# Example Swahili sentences
sentences = [
    'Jinsi ya kupika ugali?',
    'Ugali ni chakula kikuu cha Afrika Mashariki',
    'Mchele ni mazao muhimu'
]

# Generate embeddings
embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")  # [3, 384]

# Calculate similarities
similarities = model.similarity(embeddings, embeddings)
print(f"Similarities:\n{similarities}")
```

### Integration with RAG Pipeline

```python
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Use fine-tuned model for RAG
embed_model = HuggingFaceEmbedding(model_name="./test_model")

# Create vector index with enhanced embeddings
index = VectorStoreIndex.from_documents(
    documents, 
    embed_model=embed_model
)

# Query with natural language
response = index.as_query_engine().query("Jinsi ya kupika ugali?")
```

## 📚 Research Context

### Problem Statement
Current RAG systems suffer from:
1. **Knowledge base gaps** for niche topics or underrepresented languages
2. **Coverage issues** due to limited linguistic diversity in training data
3. **Formality gap** between user queries and formal knowledge bases

### Our Solution
- **Speech-derived data**: Common Voice transcriptions capture natural language variations
- **Synthetic Q&A generation**: GPT-4o creates diverse query formulations
- **Embedding fine-tuning**: Specialized training on speech-derived patterns
- **Multilingual focus**: Emphasis on underrepresented languages like Swahili

### Key Innovations
1. Repurposing speech data for text-based systems
2. Bridging formality gap through natural language patterns
3. Efficient training (only 2 epochs needed)
4. Comprehensive evaluation with IR metrics

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- **Additional Languages**: Extend approach to other languages
- **Model Improvements**: Alternative architectures or training strategies
- **Evaluation**: New metrics or benchmark datasets
- **Applications**: Real-world use cases and integrations

### Development Setup

```bash
git clone https://github.com/nickdee96/common-voice-embedding
cd common-voice-embedding
pip install -r requirements.txt
```

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{common-voice-rag-2024,
  title={Leveraging Common Voice Transcriptions for Robust Text-Based RAG: Enhancing Query Diversity Handling},
  author={THiNK (Tech Innovators Network)},
  year={2024},
  url={https://github.com/nickdee96/common-voice-embedding}
}
```

## 🏢 About THiNK

**Tech Innovators Network (THiNK)** is a Nairobi-based organization focused on digital transformation through open innovation, with particular emphasis on African languages and inclusive AI systems.

- 🌐 Website: [think.ke](https://think.ke)
- 🤗 Hugging Face: [huggingface.co/thinkKenya](https://huggingface.co/thinkKenya)
- 🐦 Follow us for updates on inclusive AI research

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mozilla Common Voice** community for providing diverse, open speech data
- **Hugging Face** ecosystem for model hosting and tools
- **BAAI** for the excellent bge-small-en base model
- All contributors to open source language technology

---

**⭐ Star this repo if you find it useful! ⭐**

For questions, issues, or collaboration opportunities, please open an issue or reach out to us directly.