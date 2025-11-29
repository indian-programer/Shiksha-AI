

import os
import pandas as pd
import streamlit as st
from openai import OpenAI

st.title("Chatbot")

# Load API Key from Streamlit Secrets
openai_api_key = st.secrets["API_KEY"]

# Create OpenAI client
client = OpenAI(api_key=openai_api_key)

# Input box
prompt = st.text_input("Ask anything:")

# On button click
if st.button("Send"):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        st.write(response.choices[0].message["content"])

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# Helper: OpenAI Chat Function
# -------------------------------
def call_openai_chat(prompt: str, temp: float = 0.2, max_t: int = 700) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_t
        )
        return resp.choices[0].message["content"].strip()

    except Exception as e:
        return f"❌ API Error: {e}"


# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Shiksha AI", layout="wide")
st.title("🎓 **Shiksha AI – Learning Assistant**")


# -------------------------------
# Sidebar Options
# -------------------------------
st.sidebar.header("Options")
mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Chat", "Upload Syllabus (CSV)", "Quiz Generator", "About"]
)

temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2)
max_tokens = st.sidebar.slider("Max Tokens (Response Length)", 100, 2000, 700, step=50)

# -------------------------------
# MODE 1: CHAT
# -------------------------------
if mode == "Chat":
    st.subheader("💬 Chat Mode")
    user_input = st.text_input("Write your question:")

    if st.button("Send", key="chat-send"):
        if user_input.strip() == "":
            st.warning("⚠️ প্রশ্ন লিখুন!")
        else:
            with st.spinner("AI উত্তর দিচ্ছে..."):
                ans = call_openai_chat(user_input, temp=temperature, max_t=max_tokens)
            st.markdown("### ✨ উত্তর")
            st.write(ans)

# -------------------------------
# MODE 2: Upload Syllabus
# -------------------------------
elif mode == "Upload Syllabus (CSV)":
    st.subheader("📂 Upload Syllabus (CSV)")
    uploaded = st.file_uploader("Upload syllabus CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ CSV পড়তে সমস্যা: {e}")
            df = pd.DataFrame()

        if not df.empty:
            st.write("📘 Uploaded Syllabus:")
            st.write(df)

            keyword = st.text_input("Search keyword:")
            if st.button("Search"):
                if keyword.strip() == "":
                    st.warning("⚠️ কীওয়ার্ড লিখুন!")
                else:
                    mask = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)
                    results = df[mask]
                    st.session_state["results"] = results
                    st.write(results)

            if "results" in st.session_state:
                results_saved = st.session_state["results"]

                if not results_saved.empty:
                    if st.button("Explain selected results with AI"):
                        combined = "\n\n".join(
                            results_saved.astype(str).apply(
                                lambda r: " | ".join(r.values.astype(str)),
                                axis=1
                            ).tolist()
                        )

                        prompt = (
                            "ছাত্রদের জন্য সহজ বাংলায় নিচের সিলেবাস পয়েন্টগুলোর ব্যাখ্যা করো:\n\n"
                            f"{combined}"
                        )

                        with st.spinner("ব্যাখ্যা তৈরি হচ্ছে..."):
                            ans = call_openai_chat(prompt, temp=temperature, max_t=max_tokens)

                        st.markdown("### 📘 ব্যাখ্যা")
                        st.write(ans)
                else:
                    st.info("🔍 কোনো ফলাফল পাওয়া যায়নি।")

# -------------------------------
# MODE 3: Quiz Generator
# -------------------------------
elif mode == "Quiz Generator":
    st.header("📝 Quick MCQ Generator")

    topic = st.text_input("বিষয় লিখুন (e.g., 'Quadratic Equations'):")
    num_q = st.slider("Number of MCQs", 1, 20, 5)
    prefer_bengali = st.checkbox("MCQ উত্তর বাংলা চাই", True)

    if st.button("Generate Quiz"):
        if topic.strip() == "":
            st.warning("⚠️ বিষয় লিখুন!")
        else:
            lang_note = "বাংলা" if prefer_bengali else "English"
            prompt = (
                f"Generate {num_q} MCQ questions on '{topic}'. "
                f"Provide options A-D. Give correct answer. Language: {lang_note}. "
                "Explain the correct answer briefly."
            )

            with st.spinner("Quiz তৈরি হচ্ছে..."):
                quiz_text = call_openai_chat(prompt, temp=temperature, max_t=max_tokens)

            st.markdown("### 📗 Generated Quiz")
            st.write(quiz_text)

            st.session_state.latest_quiz = quiz_text

    if "latest_quiz" in st.session_state:
        st.download_button(
            "Download Quiz as TXT",
            st.session_state.latest_quiz,
            file_name="quiz.txt"
        )

# -------------------------------
# MODE 4: About
# -------------------------------
elif mode == "About":
    st.header("ℹ️ Shiksha AI — Project Info")
    st.markdown("""
    **Shiksha AI** একটি অল-ইন-ওয়ান শিক্ষামূলক অ্যাপ।
    
    **Features:**
    - Chat → AI তোমার প্রশ্নের উত্তর দেয়  
    - Upload Syllabus → CSV থেকে তথ্য সার্চ ও ব্যাখ্যা  
    - Quiz Generator → যেকোনো বিষয়ে MCQ বানায়  
    """)

    st.markdown("### Quick Tips")
    st.markdown("""
    - `.env` ফাইলে `OPENAI_API_KEY=sk-xxxx` রাখুন  
    - PowerShell এ চালান: `python -m streamlit run app.py`
    - Deploy করতে GitHub + Streamlit Cloud ব্যবহার করুন  
    """)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed for **Shiksha AI**")