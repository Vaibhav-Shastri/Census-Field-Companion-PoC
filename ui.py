import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import streamlit as st

# ── Load OpenAI API Key ──
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── Load RAG artifacts ──
with open("models/embeds.pkl", "rb") as f:
    chunks, embs = pickle.load(f)
index = faiss.read_index("models/faiss.idx")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── RAG + LLM function ──
def chat_local(question: str, role: str = "enumerator") -> str:
    q_emb = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    _, idxs = index.search(q_emb, 5)
    context = [chunks[i] for i in idxs[0]]

    samples = {
        "enumerator": [
            "What if a house is locked?",
            "How to record a vacant dwelling?"
        ],
        "supervisor": [
            "How do I handle inaccessible households?",
            "When should I flag survey inconsistencies?"
        ]
    }[role]

    prompt = (
        f"You are Census Field Companion (role: {role}).\n"
        f"Sample questions: {samples}\n"
        "Answer using only these excerpts (cite heading):\n\n"
    )
    for c in context:
        prompt += f"[{c['heading']}] {c['text']}\n"
    prompt += f"\nUser: {question}\nAnswer:"

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

# ── Streamlit UI ──
st.set_page_config(page_title="Census Field Companion", layout="wide")
hide_style = """
    <style>
      #MainMenu {visibility: hidden !important;}
      footer {visibility: hidden !important;}
      button[aria-label="Open navigation"] {visibility: hidden !important;}
      a[href*="github.com"] {visibility: hidden !important;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🗄️ Manuals Used (from website)")
for m in [
    "HouseListing_Housing_Census_2011.pdf",
    "Abridged_Houselist_Household_Schedule.pdf",
    "Supervisor_Handbook_Flowcharts.pdf",
    "Household_Schedule_Manual.pdf",
    "Houselisting_Housing_Census_Schedule.pdf",
    "Urban_Frame_Jurisdiction/*"
]:
    st.sidebar.write(f"- {m}")
st.sidebar.markdown("""
**Helps**  
- Enumerators with SOP guidance  
- Supervisors manage field issues  

**Powered by**  
GPT-3.5 Turbo  
""")

# Main area
st.title("📡 PoC: Census Field Companion for ORGI")
st.markdown("Select your role, review sample questions, and ask your question below.")

role = st.selectbox("👤 Your Role", ["enumerator", "supervisor"])
st.markdown("**Sample questions:**")
for q in {
    "enumerator": ["What if a house is locked?", "How to record a vacant dwelling?"],
    "supervisor": [
        "How do I handle inaccessible households?",
        "When should I flag survey inconsistencies?"
    ]
}[role]:
    st.write(f"- {q}")

query = st.text_input("💬 Ask your question")
if st.button("Submit"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Getting answer…"):
            answer = chat_local(query, role)
        st.markdown("**Answer:**")
        st.write(answer)
