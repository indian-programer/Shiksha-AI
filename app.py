# app.py
import os
import streamlit as st
import pandas as pd
from typing import Optional, List

# Try to create an OpenAI client that works with different openai versions:
def make_openai_client(api_key: str):
    """
    Tries different import styles so this code works with multiple openai package variants.
    Returns a tuple (client_obj, style) where style is "new" or "legacy".
    """
    try:
        # New official SDK style (openai >= 1.0 with OpenAI class)
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        return client, "new"
    except Exception:
        pass

    try:
        # Legacy style (openai package that exposes methods on module)
        import openai  # type: ignore
        openai.api_key = api_key
        return openai, "legacy"
    except Exception:
        pass

    return None, None

# Read API key: prefer st.secrets, else environment var
def read_api_key() -> Optional[str]:
    key = None
    try:
        # common names: OPENAI_API_KEY or API_KEY
        if "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
        elif "API_KEY" in st.secrets:
            key = st.secrets["API_KEY"]
    except Exception:
        # st.secrets might not exist locally
        pass

    if not key:
        # environment fallback
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    return key

API_KEY = read_api_key()
if not API_KEY:
    st.warning("🔑 OpenAI API key পাওয়া যায়নি। লোকালি ~/.streamlit/secrets.toml এ `OPENAI_API_KEY = \"sk-...\"` যোগ করুন অথবা Streamlit Cloud secrets এ সেট করুন.")
    # still continue but API calls will fail with nice message

client_obj, client_style = (None, None)
if API_KEY:
    client_obj, client_style = make_openai_client(API_KEY)

# Helper that works with both styles
def call_openai_chat(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.2, max_tokens: int = 600) -> str:
    if not API_KEY or not client_obj:
        return "OpenAI API key নেই — st.secrets['OPENAI_API_KEY'] বা environment variable সেট করুন."

    try:
        if client_style == "new":
            # new SDK: client.chat.completions.create(...)
            resp = client_obj.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # new SDK shape: resp.choices[0].message["content"]
            text = ""
            try:
                text = resp.choices[0].message["content"]
            except Exception:
                # sometimes different attr names
                text = str(resp)
            return text.strip()
        elif client_style == "legacy":
            # legacy openai package (openai.ChatCompletion.create)
            resp = client_obj.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # legacy shape: resp.choices[0].message['content'] or resp.choices[0].text
            try:
                return resp.choices[0].message["content"].strip()
            except Exception:
                try:
                    return resp.choices[0].text.strip()
                except Exception:
                    return str(resp)
        else:
            return "OpenAI ক্লায়েন্ট ইনিশিয়ালাইজ হয়নি (unsupported)."
    except Exception as e:
        return f"OpenAI call failed: {e}"

# Streamlit app UI
st.set_page_config(page_title="Shiksha AI", layout="wide")
st.title("🔮 Shiksha AI — Learning Assistant")

# Sidebar
st.sidebar.header("Options")
mode = st.sidebar.selectbox("Mode", ["Chat", "Upload Syllabus (CSV)", "Quiz Generator", "About"])

temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2, step=0.1)
max_tokens = st.sidebar.slider("Max tokens (response length)", 100, 1500, 700, step=50)

if mode == "Chat":
    st.subheader("💬 Chat Mode")
    st.write("Ask questions in Bengali or English. The assistant will reply using OpenAI.")
    user_input = st.text_input("প্রশ্ন লিখুন:", key="chat_input")
    if st.button("Send", key="chat_send"):
        if not user_input or user_input.strip() == "":
            st.warning("প্রশ্ন লিখুন!")
        else:
            with st.spinner("AI উত্তর তৈরি করছে..."):
                prompt = user_input.strip()
                ans = call_openai_chat(prompt, temperature=temperature, max_tokens=max_tokens)
                st.markdown("### উত্তর")
                st.write(ans)

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
                    combined = "\n\n".join(results_saved.astype(str).apply(lambda r: " | ".join(r.values.astype(str)), axis=1).tolist())
                    prompt = f"ছাত্রদের জন্য সহজ বাংলায় নিচের বিষয়টি ব্যাখ্যা করো:\n\n{combined}"
                    with st.spinner("ব্যাখ্যা তৈরি হচ্ছে..."):
                        ans = call_openai_chat(prompt, temperature=temperature, max_tokens=max_tokens)
                        st.markdown("### ব্যাখ্যা")
                        st.write(ans)
        else:
            st.info("ফাইল পড়া যায়নি বা ফাইল খালি — সঠিক CSV আপলোড করুন।")

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

elif mode == "About":
    st.header("About — Shiksha AI")
    st.markdown(
        """
- A lightweight Streamlit learning assistant for students.
- Modes: Chat (with OpenAI), Upload & Search syllabus (CSV), Quiz generator.
- Make sure `~/.streamlit/secrets.toml` or Streamlit Cloud secrets contains:

"""
    )

# Footer
st.markdown("---")
st.caption("Developed for Shiksha AI — If you want, provide a sample syllabus CSV & requirements.txt and I'll help fine-tune the app.")
