**Topic:**  
_Leveraging Common Voice Transcriptions for Robust Text-Based RAG: Enhancing Query Diversity Handling_

**Refined Abstract:**  
This work investigates the strategic integration of Mozilla Common Voice—a multilingual, crowd-sourced speech dataset—to address two critical limitations in text-based Retrieval-Augmented Generation (RAG) systems: **handling knowledge base gaps** and **mitigating coverage especially for niche topics or underrepresented languages.** While RAG systems traditionally rely on written text corpora, we propose that Common Voice’s _speech transcriptions_—with their inherent linguistic diversity and speaker variability—offer unique value for improving robustness in real-world applications.

We introduce a pipeline that (1) fine-tunes query-encoding/embedding models using Common Voice transcriptions to better interpret paraphrased, ambiguous, and (2) augments knowledge bases with relevant subsets of transcribed speech data (e.g., terms in non-English contexts). The system employs semantic search over a hybrid knowledge base (original documents + curated Common Voice subsets) and generates responses via a Large Language Model (LLM).

To evaluate improvements, we design metrics emphasizing **query diversity robustness** (success rate on rephrased/vernacular queries) and **coverage breadth** (response accuracy for underrepresented topics), alongside standard relevance/factuality scoring. Experiments compare baseline RAG performance against our Common Voice-enhanced variant, demonstrating how speech-derived text data bridges the "formality gap" between curated knowledge bases and real-world user interactions.

This work highlights Hugging Face’s Common Voice as a resource not just for speech tasks, but for strengthening _text-based_ NLP systems through its linguistically rich, naturally variable transcriptions. We further release code for preprocessing Common Voice data into RAG-compatible formats and provide guidelines for optimizing speech-to-text augmentation in retrieval systems.

**Key Value Proposition:**  
By repurposing Common Voice’s speech data for text-based RAG, we enable systems to better align with how users _actually speak_ rather than how they write, while expanding access to non-dominant languages and domains underrepresented in conventional text corpora.

**Session Format: A 2 hour workshop that showcases practically some of the key steps to students and participants.**

**Workshop Title:** _"Building Linguistically Robust RAG Systems with Common Voice Transcriptions"_

**Target Audience**

- NLP engineers, data scientists, or students/researchers familiar with RAG basics (retrieval, LLMs).
- Intermediate Python/PyTorch/Hugging Face proficiency.