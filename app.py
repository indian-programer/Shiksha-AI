        print("ChatCompletion error or unavailable:", e)

    # Fallback to older Completion API (text-davinci-003)
    try:
        if hasattr(openai, "Completion"):
            resp = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=max_t,
                temperature=temp,# app.py
import os
import streamlit as st
from dotenv import load_dotenv
import openai
import pandas as pd
from typing import List, Dict

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# safe import of new OpenAI SDK OpenAI class
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    print("openai import failed:", e)

from openai import OpenAI
import os

def call_openai_chat(prompt: str, temp: float = 0.2, max_t: int = 700) -> str:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=max_t,
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        return f"OpenAI API error — API key / internet / library check করুন।\nError: {str(e)}"

# --- END replace block ---

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Streamlit runs top-to-bottom and st.warning here is fine
    st.warning("OPENAI_API_KEY পাওয়া যায়নি। চালাতে .env ফাইলে OPENAI_API_KEY=sk-... রাখো।")
else:
    # set api key for openai library
    openai.api_key = OPENAI_API_KEY

# Helper: call OpenAI (tries ChatCompletion first, falls back to Completion)
def call_openai_chat(prompt: str, temp: float = 0.2, max_t: int = 700) -> str:
    """
    Try ChatCompletion (gpt-3.5-turbo) if available; otherwise fall back to Completion (text-davinci-003).
    Returns the assistant text (string).
    """
    try:
        # Try ChatCompletion (works on openai versions that support it)
        if hasattr(openai, "ChatCompletion"):
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_t,
            )
            # extract text
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # fallback below

            )
            return resp["choices"][0]["text"].strip()
    except Exception as e:
        print("Completion fallback error:", e)

    # If both failed, return an error message
    return "OpenAI API call failed — চেক করো API key এবং openai লাইব্রেরি ভার্সন।"

# Streamlit page config
st.set_page_config(page_title="Shiksha AI", layout="wide")
st.title("🔎 Shiksha AI — সহায়ক চ্যাটবট")

# Sidebar options
st.sidebar.header("Options")
mode = st.sidebar.selectbox("Mode", ["Chat", "Upload Syllabus (CSV)", "Quiz Generator", "About"])
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2)
max_tokens = st.sidebar.slider("Max tokens (response length)", 100, 1500, 700, step=50)

# Mode: Chat
if mode == "Chat":
    st.subheader("💬 Chat Mode")
    user_input = st.text_input("Write your question:", key="chat_input")
    if st.button("Send", key="chat_send"):
        if not user_input.strip():
            st.warning("প্রশ্ন লিখুন।")
        else:
            with st.spinner("AI উত্তর দিয়ে থাকছে..."):
                ans = call_openai_chat(user_input, temp=temperature, max_t=max_tokens)
            st.markdown("### উত্তর")
            st.write(ans)

# Mode: Upload Syllabus
elif mode == "Upload Syllabus (CSV)":
    st.subheader("📚 Upload Syllabus (CSV)")
    uploaded = st.file_uploader("Upload syllabus CSV", type=["csv"], key="uploader")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"CSV পড়তে সমস্যা: {e}")
            df = pd.DataFrame()
        if not df.empty:
            st.write(df)

            keyword = st.text_input("Search keyword:", key="search_keyword")
            # mask will look for keyword in any column (case-insensitive)
            if keyword.strip() != "":
                mask = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)
            else:
                mask = pd.Series([False] * len(df))

            if st.button("Search in syllabus", key="search_syllabus"):
                results = df[mask]
                st.session_state["results"] = results
                st.write(results)

            results_saved = st.session_state.get("results", None)
            if results_saved is not None and not results_saved.empty:
                if st.button("Explain selected results (with AI)", key="explain_selected"):
                    combined = "\n\n".join(
                        results_saved.astype(str).apply(lambda r: " | ".join(r.values.astype(str)), axis=1).tolist()
                    )
                    prompt = f"ছাত্রদের জন্য সহজ বাংলায় নিচের বিষয়টি ব্যাখ্যা করো:\n\n{combined}"
                    with st.spinner("ব্যাখ্যা তৈরি হচ্ছে..."):
                        ans = call_openai_chat(prompt, temp=temperature, max_t=max_tokens)
                    st.markdown("### ব্যাখ্যা")
                    st.write(ans)
            else:
                st.info("প্রথমে 'Search in syllabus' চালাও এবং ফলাফল দেখো।")
        else:
            st.info("ভিতরে কোনো ডাটা পাওয়া যায়নি — সঠিক CSV আপলোড করো।")

# Mode: Quiz Generator
elif mode == "Quiz Generator":
    st.header("Quick MCQ Generator")
    topic = st.text_input("বিষয়/টপিক (e.g., 'Quadratic Equations')", key="quiz_topic")
    num_q = st.slider("Number of MCQs", 1, 10, 5, key="num_q")
    prefer_bengali = st.checkbox("উত্তর বাংলায় চাই", value=True, key="prefer_bengali")

    if st.button("Generate Quiz", key="generate_quiz"):
        if topic.strip() == "":
            st.warning("অনুগ্রহ করে একটি টপিক লিখো।")
        else:
            lang_note = "বাংলা" if prefer_bengali else "English"
            prompt = (
                f"Generate {num_q} multiple choice questions for students on the topic '{topic}'. "
                f"Provide each question, 4 options labelled A-D, and indicate the correct option letter. "
                f"Keep language simple ({lang_note}). Also include short explanation for the correct answer."
            )
            with st.spinner("Quiz তৈরি হচ্ছে..."):
                quiz_text = call_openai_chat(prompt, temp=temperature, max_t=max_tokens)
            st.markdown("### Generated Quiz")
            st.write(quiz_text)
            st.session_state.latest_quiz = quiz_text

    if "latest_quiz" in st.session_state:
        st.download_button("Download Quiz as TXT", st.session_state.latest_quiz, file_name="quiz.txt")

# Mode: About
elif mode == "About":
    st.header("Shiksha AI — Project Info")
    st.markdown(
        """
- এটি একটি **MVP** — ছাত্রদের প্রশ্নের উত্তর, সিলেবাস অনুসন্ধান এবং কুইজ জেনারেট করতে সক্ষম।
- চালাতে: `.env` ফাইলে `OPENAI_API_KEY=sk-...` রাখো, এবং `pip install -r requirements.txt` করে `python -m streamlit run app.py` চালাও।
- যদি `app.py` ফাইলটি 0 byte হয়ে থাকে, এই কোডটি পেস্ট করে সেভ করে দাও।
"""
    )
    st.markdown("### Quick tips")
    st.markdown(
        """
- PowerShell-এ space ছিল এমন path-এ যেতে: `cd \"C:\\Folders\\Shiksha AI\"`
- Deploy করতে চাইলে GitHub-এ পুশ করে Streamlit Community Cloud ব্যবহার করো।
"""
    )

# Footer
st.markdown("---")
st.caption("Developed for Shiksha AI — যদি চাও আমি sample syllabus CSV ও requirements.txt দিয়েও দিতে পারি।")
