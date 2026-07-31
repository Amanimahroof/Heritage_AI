"""
app.py
------
Streamlit App + Conversation Memory

What this does:
1. Lets the user upload an image of a building
2. Runs your CV model -> shows predicted style + confidence
3. Provides a chat box where the user can ask follow-up questions
4. Uses a self-built SimpleConversationMemory class so the LLM remembers earlier
   turns in the same session (multi-turn conversation, 3+ exchanges required by
   the brief). LangChain's own ConversationBufferMemory was tried first but
   caused a ModuleNotFoundError due to version changes, so this lightweight
   equivalent replaced it.
5. Every answer includes citations from your RAG knowledge base

How to run:
    (heritage_env) > cd RAG_System
    (heritage_env) > streamlit run app.py

This opens in your browser automatically (usually http://localhost:8501)

Note on Streamlit's execution model (the "fiddly" part mentioned in chat):
Streamlit re-runs this ENTIRE script top-to-bottom every time you interact
with any widget (upload a file, type a message, click a button). To avoid
reloading the CV model / vector store / chat history every single time,
we use st.session_state (persists across reruns) and @st.cache_resource
(loads expensive things once, reuses them after).
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Simple self-contained memory (avoids relying on langchain.memory, which has
# moved/changed across langchain versions and caused a ModuleNotFoundError).
# Does the same job: stores turns, formats them as a string for the prompt.
class SimpleConversationMemory:
    def __init__(self):
        self.turns = []  # list of (question, answer) tuples

    def save_context(self, inputs, outputs):
        self.turns.append((inputs["input"], outputs["output"]))

    @property
    def buffer_as_str(self):
        if not self.turns:
            return "(no previous conversation)"
        lines = []
        for question, answer in self.turns:
            lines.append(f"User: {question}")
            lines.append(f"Assistant: {answer}")
        return "\n".join(lines)

load_dotenv()

# ---- CONFIG ----
MODEL_PATH = os.path.join("..", "Models", "efficientnet_heritage.pth")
CLASS_NAMES = ["Baroque", "Gothic", "Neoclassical", "Roman", "Victorian"]
VECTORSTORE_DIR = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 6

CHAT_PROMPT_TEMPLATE = """You are a heritage architecture assistant for the National Heritage Preservation Trust.
{cv_context}
Using ONLY the reference context below, answer the user's question. If context
is insufficient, say so honestly rather than guessing.

Conversation so far:
{chat_history}

Reference context:
{context}

User question: {question}

After your answer, list sources used in this format:
Sources: [filename1, filename2, ...]

Answer:"""


# ---------- Cached resource loaders (run once, not on every rerun) ----------

@st.cache_resource
def load_cv_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{EMBEDDING_MODEL}")
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def load_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0.2)


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_style(model, image):
    transform = get_transform()
    image_tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    return CLASS_NAMES[predicted_idx.item()], confidence.item()


def format_context_with_sources(retrieved_docs):
    context_parts, citations = [], []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[{source}, page {page}]\n{doc.page_content}")
        citations.append(source)
    seen = set()
    unique_citations = [c for c in citations if not (c in seen or seen.add(c))]
    return "\n\n---\n\n".join(context_parts), unique_citations


def ask_question(question, vectorstore, llm, memory, cv_prediction=None):
    retrieved_docs = vectorstore.similarity_search(question, k=TOP_K)
    context, citations = format_context_with_sources(retrieved_docs)

    cv_context = ""
    if cv_prediction:
        style, confidence = cv_prediction
        cv_context = f"An uploaded image was classified as {style} architecture with {confidence:.1%} confidence.\n"

    chat_history_text = memory.buffer_as_str

    prompt = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)
    messages = prompt.format_messages(
        cv_context=cv_context,
        chat_history=chat_history_text,
        context=context,
        question=question,
    )

    response = llm.invoke(messages)
    memory.save_context({"input": question}, {"output": response.content})
    return response.content, citations


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Heritage AI", page_icon="🏛️", layout="wide")

# Custom styling — visual only, no functional changes
st.markdown("""
<style>
.main .block-container { padding-top: 2rem; max-width: 1100px; }
.hero {
    background: linear-gradient(135deg, #4a3f6b 0%, #7c5c8a 50%, #b3763d 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
}
.hero h1 { color: white; margin: 0; font-size: 2rem; }
.hero p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 1rem; }
.section-card {
    background: #ffffff; border: 1px solid #e8e4f0; border-radius: 14px;
    padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.style-badge {
    display: inline-block; padding: 0.4rem 1rem; border-radius: 20px;
    background: #4a3f6b; color: white; font-weight: 600; font-size: 1.1rem;
}
.confidence-bar-bg { background: #eee; border-radius: 8px; height: 14px; margin-top: 0.5rem; overflow: hidden; }
.confidence-bar-fill { height: 100%; border-radius: 8px; transition: width 0.3s ease; }
</style>
""", unsafe_allow_html=True)

STYLE_ICONS = {
    "Gothic": "⛪", "Baroque": "🎭", "Neoclassical": "🏛️",
    "Roman": "🏟️", "Victorian": "🏘️",
}

st.markdown("""
<div class="hero">
    <h1>🏛️ Heritage AI — Architectural Style Assistant</h1>
    <p>National Heritage Preservation Trust prototype · CV classification + cited RAG chat</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state (persists across Streamlit reruns)
if "memory" not in st.session_state:
    st.session_state.memory = SimpleConversationMemory()
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # list of (role, text) for display
if "cv_prediction" not in st.session_state:
    st.session_state.cv_prediction = None

# --- Image upload + CV prediction ---
left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 📤 1. Upload a building image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        cv_model = load_cv_model()
        style, confidence = predict_style(cv_model, image)
        st.session_state.cv_prediction = (style, confidence)

        icon = STYLE_ICONS.get(style, "🏗️")
        bar_color = "#2e7d32" if confidence >= 0.8 else ("#e6a817" if confidence >= 0.6 else "#c0392b")

        st.markdown(f"""
        <div style="margin-top:1rem;">
            <span class="style-badge">{icon} {style}</span>
            <div style="margin-top:0.6rem; font-size:0.9rem; color:#555;">Confidence: <b>{confidence:.1%}</b></div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{confidence*100}%; background:{bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < 0.6:
            st.warning("Confidence is fairly low — treat this prediction with caution.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Chat interface ---
with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 💬 2. Ask questions")
    st.caption("E.g. 'What defines this style?' or 'How does it compare to Baroque?'")

    vectorstore = load_vectorstore()
    llm = load_llm()

    chat_container = st.container(height=380)
    with chat_container:
        for role, text in st.session_state.chat_log:
            avatar = "🧑" if role == "user" else "🏛️"
            with st.chat_message(role, avatar=avatar):
                st.write(text)

    user_question = st.chat_input("Ask about the detected style, or anything in the knowledge base...")

    if user_question:
        st.session_state.chat_log.append(("user", user_question))
        with chat_container:
            with st.chat_message("user", avatar="🧑"):
                st.write(user_question)

            with st.chat_message("assistant", avatar="🏛️"):
                with st.spinner("Thinking..."):
                    answer, citations = ask_question(
                        user_question,
                        vectorstore,
                        llm,
                        st.session_state.memory,
                        cv_prediction=st.session_state.cv_prediction,
                    )
                    st.write(answer)
                    st.caption(f"📚 Sources: {', '.join(citations)}")

        st.session_state.chat_log.append(("assistant", f"{answer}\n\n*Sources: {', '.join(citations)}*"))
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# --- Reset button ---
if st.button("🔄 Reset conversation"):
    st.session_state.memory = SimpleConversationMemory()
    st.session_state.chat_log = []
    st.session_state.cv_prediction = None
    st.rerun()
