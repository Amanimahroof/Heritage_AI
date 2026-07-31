# Heritage AI — Architectural Style Classification & RAG Assistant

A prototype AI system for the National Heritage Preservation Trust (NHPT), combining
computer vision (architectural style classification) with a LangChain RAG conversational
assistant, developed for NB627BSDS (Machine Learning and Related Applications).

## Project Structure

```
heritage-ai/
├── CV_Model/
│   ├── CV_training.ipynb       # Full CV training pipeline (Google Colab): dataset loading,
│   │                           # EfficientNetB0 transfer learning, evaluation, Grad-CAM
│   └── test_model_load.py      # Confirms the trained model loads and runs on CPU (laptop)
│
├── RAG_System/
│   ├── RAG_Pipeline.ipynb      # Full RAG pipeline: document loading, vector store, RAG
│   │                           # chain with citations, CV→LLM integration, multi-turn
│   │                           # conversation with memory (6 example conversations)
│   └── app.py                  # Streamlit application layer — interactive UI for image
│                                # upload, style prediction, and chat
│
├── Models/
│   └── efficientnet_heritage.pth   # Trained CV model weights (not tracked if >100MB —
│                                    # see note below)
│
├── Documents/                  # 6 heritage architecture reference documents (PDF), used
│                                # as the RAG knowledge base: Gothic, Baroque, Romanesque,
│                                # Neoclassical, Victorian/Queen Anne, Conservation
│
├── .gitignore
└── README.md
```

## Setup

1. Create and activate a virtual environment:
```
python -m venv heritage_env
heritage_env\Scripts\activate      # Windows
```

2. Install dependencies:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install langchain langchain-community langchain-text-splitters langchain-openai
pip install faiss-cpu sentence-transformers pypdf streamlit pillow python-dotenv ipykernel
```

3. Create a `.env` file inside `RAG_System/` with your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```
(Not included in this repo — you'll need your own key to run the LLM-dependent cells)

## How to Run

**CV training (already run, results saved in the notebook):**
Open `CV_Model/CV_training.ipynb` — this was run on Google Colab with GPU. Outputs
(accuracy, confusion matrix, Grad-CAM) are saved inline in the notebook.

**Confirm CV model loads locally:**
```
cd CV_Model
python test_model_load.py
```

**RAG pipeline (documents → vector store → RAG chain → CV integration → multi-turn chat):**
Open `RAG_System/RAG_Pipeline.ipynb` in VS Code or Jupyter (kernel: `heritage_env`), and
run all cells top to bottom. Sections 1-2 build and save the vector store (only needs
running once); Sections 3-5 demonstrate retrieval, citations, CV-LLM integration, and
multi-turn conversation memory.

**Interactive Streamlit app:**
```
cd RAG_System
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`. Upload a building image to get a
style prediction, then chat with the assistant — it remembers earlier questions in
the same session.

## System Overview

1. A user uploads an image of a building
2. An EfficientNetB0 model (transfer learning, 92% test accuracy across 5 architectural
   styles: Baroque, Gothic, Neoclassical, Roman, Victorian) predicts the style with a
   confidence score
3. That structured prediction ({style}, {confidence}) is handed off to a RAG pipeline,
   which retrieves relevant chunks from 6 heritage documents (FAISS vector store,
   all-MiniLM-L6-v2 embeddings)
4. GPT-4o-mini generates a grounded, cited explanation using the retrieved context
5. A self-built lightweight memory component tracks conversation history, enabling
   multi-turn follow-up questions

## Note on Model File Size

If `efficientnet_heritage.pth` exceeds GitHub's 100MB limit, it is linked via Google
Drive instead of committed directly: [add your Drive link here if applicable]
