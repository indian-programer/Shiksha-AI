# app.py
import os
from typing import Optional
import pandas as pd
import streamlit as st
from openai import OpenAI

# Page config
st.set_page_config(page_title="Shiksha AI", layout="wide")

# Title
st.title("🔮 Shiksha AI — Learning Assistant")

# -------------------------
# Helper: read API key
# -------------------------
def read_api_key() -> Optional[str]:
    """Try st.secrets first, then environment variables."""
    key = None
    try:
        # Try common keys in st.secrets (works on Streamlit Cloud / local secrets)
        if "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
        elif "API_KEY" in st.secrets:
            key = st.secrets["API_KEY"]
    except Exception:
        # st.secrets might not exist locally
        pass

    if not key:
        # Fallback to environment variables
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")

    return key

API_KEY = read_api_key()
if not API_KEY:
    st.warning(
        "🔑 OpenAI API key পাওয়া যায়নি।\n"
        "Streamlit Cloud-এ Secrets এ `OPENAI_API_KEY` বা `API_KEY` যোগ করুন বা লোকালি ~/.streamlit/secrets.toml এ সেট করুন."
    )

# -------------------------
# Create OpenAI client (new SDK)
# -------------------------
client: Optional[OpenAI] = None
if API_KEY:
    try:
        client = OpenAI(api_key=API_KEY)
    except Exception as e:
        st.error(f"OpenAI ক্লায়েন্ট ইনিশিয়ালাইজ করা যায়নি: {e}")
        client = None

# -------------------------
# Helper: call OpenAI chat
# -------------------------
def call_openai_chat(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> str:
    """Call OpenAI (new SDK). Return assistant text or error message string."""
    if not client:
        return "OpenAI API key সেট নেই — st.secrets বা environment এ সেট করুন."

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Best-effort extraction of assistant text
        try:
            return resp.choices[0].message["content"].strip()
        except Exception:
            # fallback to str(resp)
            return str(resp)
    except Exception as e:
        return f"OpenAI call failed: {e}"

# -------------------------
# Sidebar / controls
# -------------------------
st.sidebar.header("Options")
mode = st.sidebar.selectbox("Mode", ["Chat", "Upload Syllabus (CSV)", "Quiz Generator", "About"])

temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2, step=0.1)
max_tokens = st.sidebar.slider("Max tokens (response length)", 100, 1500, 700, step=50)

# -------------------------
# Mode: Chat
# -------------------------
if mode == "Chat":
    st.subheader("💬 Chat Mode")
    st.write("Ask questions in Bengali or English. The assistant will reply using OpenAI.")
    user_input = st.text_input("প্রশ্ন লিখুন:", key="chat_input")
    if st.button("Send", key="chat_send"):
        if not user_input or user_input.strip() == "":
            st.warning("প্রশ্ন লিখুন!")
        else:
            with st.spinner("AI উত্তর তৈরি করছে..."):
                prompt_text = user_input.strip()
                ans = call_openai_chat(prompt_text, temperature=temperature, max_tokens=max_tokens)
                st.markdown("### উত্তর")
                st.write(ans)

# -------------------------
# Mode: Upload syllabus (CSV)
# -------------------------
elif mode == "Upload Syllabus (CSV)":
    st.subheader("📄 Upload syllabus (CSV) — Searchable")
    uploaded = st.file_uploader("Upload syllabus CSV", type=["csv"], key="uploader")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"CSV পড়তে সমস্যা: {e}")
            df = pd.DataFrame()
        if not df.empty:
            st.write("Preview:", df.head(20))
            keyword = st.text_input("Search keyword:", key="search_keyword")
            if keyword and keyword.strip() != "":
                mask = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)
            else:
                mask = pd.Series([False] * len(df))
            if st.button("Search in syllabus", key="search_syllabus"):
                results = df[mask]
                st.write(results)
                st.session_state["results"] = results
            else:
                st.info("অনুসন্ধান চালাতে 'Search in syllabus' চাপুন")
            results_saved = st.session_state.get("results", None)
            if results_saved is not None and not results_saved.empty:
                if st.button("Explain selected results (with AI)", key="explain_selected"):
                    combined = "\n\n".join(
                        results_saved.astype(str).apply(lambda r: " | ".join(r.values.astype(str)), axis=1).tolist()
                    )
                    prompt = f"ছাত্রদের জন্য সহজ বাংলায় নিচের বিষয়টি ব্যাখ্যা করো:\n\n{combined}"
                    with st.spinner("ব্যাখ্যা তৈরি হচ্ছে..."):
                        ans = call_openai_chat(prompt, temperature=temperature, max_tokens=max_tokens)
                        st.markdown("### ব্যাখ্যা")
                        st.write(ans)
        else:
            st.info("ফাইল পড়া যায়নি বা ফাইল খালি — সঠিক CSV আপলোড করুন।")

# -------------------------
# Mode: Quiz Generator
# -------------------------
elif mode == "Quiz Generator":
    st.subheader("📝 Quick MCQ Generator")
    topic = st.text_input("বিষয়/টপিক (e.g., Quadratic Equations)", key="quiz_topic")
    num_q = st.slider("Number of MCQs", 1, 20, 5, key="num_q")
    prefer_bengali = st.checkbox("উত্তর বাংলায় চাই", value=True, key="prefer_bengali")
    if st.button("Generate Quiz", key="generate_quiz"):
        if not topic or topic.strip() == "":
            st.warning("অনুগ্রহ করে একটি টপিক লিখুন।")
        else:
            lang_note = "বাংলা" if prefer_bengali else "English"
            prompt = (
                f"Generate {num_q} multiple choice questions for students on the topic '{topic}'. "
                f"Provide each question, 4 options labelled A-D, and indicate the correct option letter. "
                f"Keep language simple ({lang_note}). Also include a one-line explanation for each correct answer."
            )
            with st.spinner("Quiz তৈরী হচ্ছে..."):
                quiz_text = call_openai_chat(prompt, temperature=temperature, max_tokens=max_tokens)
                st.markdown("### Generated Quiz")
                st.write(quiz_text)
                st.session_state["latest_quiz"] = quiz_text
    if "latest_quiz" in st.session_state:
        st.download_button("Download Quiz as TXT", st.session_state["latest_quiz"], file_name="quiz.txt")

# -------------------------
# Mode: About
# -------------------------
elif mode == "About":
    st.header("About — Shiksha AI")
    st.markdown(
        """
- A lightweight Streamlit learning assistant for students.
- Modes: Chat (with OpenAI), Upload & Search syllabus (CSV), Quiz generator.

**Make sure your API key is set**:
- On Streamlit Cloud: go to App → Settings → Secrets and add `OPENAI_API_KEY = "sk-..."` (or `API_KEY`).
- Locally: create `~/.streamlit/secrets.toml` with:

"""
    )

# Footer
st.markdown("---")
st.caption("Developed for Shiksha AI — provide a sample syllabus CSV & requirements.txt if you want further help.")
