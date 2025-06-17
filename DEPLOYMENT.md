# Deployment Guide

This guide covers different ways to deploy and use the fine-tuned Common Voice embedding model.

## 🚀 Quick Deployment Options

### 1. Local Development

```bash
# Clone and setup
git clone https://github.com/nickdee96/common-voice-embedding.git
cd common-voice-embedding
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run the notebook
jupyter notebook finetune_embedding.ipynb
```

### 2. Docker Deployment

```bash
# Build the Docker image
docker build -t common-voice-rag .

# Run the container
docker run -p 8888:8888 -v $(pwd):/workspace common-voice-rag
```

### 3. Cloud Deployment

#### Google Colab
1. Upload the notebook to Google Colab
2. Install requirements in the first cell
3. Upload your .env file or set environment variables
4. Run all cells

#### AWS SageMaker
1. Create a new SageMaker notebook instance
2. Upload the project files
3. Install requirements
4. Execute the training pipeline

## 🔧 Production Integration

### Using the Fine-tuned Model

```python
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load the fine-tuned model
embed_model = HuggingFaceEmbedding(model_name="./test_model")

# Create RAG system
index = VectorStoreIndex.from_documents(
    documents, 
    embed_model=embed_model
)

# Query the system
query_engine = index.as_query_engine()
response = query_engine.query("Your question here")
```

### API Deployment

```python
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("./test_model")

@app.post("/embed")
async def embed_text(text: str):
    embedding = model.encode([text])
    return {"embedding": embedding.tolist()}

@app.post("/similarity")
async def compute_similarity(text1: str, text2: str):
    embeddings = model.encode([text1, text2])
    similarity = model.similarity(embeddings, embeddings)
    return {"similarity": float(similarity[0][1])}
```

Run with:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## 📊 Performance Considerations

### Memory Requirements
- **Training**: 8GB+ GPU memory recommended
- **Inference**: 2GB+ GPU memory (or CPU with slower performance)
- **Model Size**: ~133MB for the fine-tuned model

### Speed Optimization
```python
# Use GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("./test_model", device=device)

# Batch processing for better throughput
texts = ["text1", "text2", "text3", ...]
embeddings = model.encode(texts, batch_size=32)
```

### Scaling Considerations
- Use model quantization for faster inference
- Consider ONNX conversion for production deployment
- Implement caching for frequently accessed embeddings

## 🔒 Security and Privacy

### Environment Variables
Never commit API keys or sensitive data. Use environment variables:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### Data Privacy
- Common Voice data is public domain
- Generated Q&A pairs may contain sensitive information
- Implement appropriate data handling procedures

## 🌍 Multi-language Deployment

### Language-specific Models
Train separate models for different languages:
```bash
# Train for different languages
python train.py --language sw  # Swahili
python train.py --language yo  # Yoruba
python train.py --language ha  # Hausa
```

### Language Detection
```python
from langdetect import detect

def route_to_model(text):
    lang = detect(text)
    if lang == "sw":
        return swahili_model.encode([text])
    elif lang == "en":
        return english_model.encode([text])
    else:
        return default_model.encode([text])
```

## 📈 Monitoring and Evaluation

### Performance Metrics
```python
# Track key metrics
metrics = {
    "accuracy_at_1": 0.924,
    "accuracy_at_5": 0.959,
    "ndcg_at_10": 0.944,
    "mrr_at_10": 0.938
}

# Log to monitoring system
import wandb
wandb.log(metrics)
```

### A/B Testing
```python
import random

def get_embedding_model(user_id):
    # Route traffic between models
    if hash(user_id) % 10 < 5:  # 50% traffic
        return enhanced_model
    else:
        return baseline_model
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   batch_size = 4  # Instead of 10
   ```

2. **Model Loading Errors**
   ```python
   # Check model path
   import os
   assert os.path.exists("./test_model/config.json")
   ```

3. **API Rate Limits**
   ```python
   # Add retry logic
   import time
   from openai import RateLimitError
   
   try:
       response = openai_client.chat.completions.create(...)
   except RateLimitError:
       time.sleep(60)  # Wait before retry
   ```

## 📞 Support

For deployment issues:
- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Open an issue on GitHub
- Contact us at [contact@think.ke]

---

Happy deploying! 🚀
