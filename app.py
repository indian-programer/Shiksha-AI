# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import openai
import pandas as pd
from typing import List, Dict

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY পাওয়া যায়নি। চালাতে .env ফাইলে OPENAI_API_KEY=sk-... রাখো।")
else:
    openai.api_key = OPENAI_API_KEY

# Streamlit page config
st.set_page_config(page_title="Shiksha AI — Class 10", layout="wide")
st.title("🔎 Shiksha AI — Class 10 সহায়ক চ্যাটবট")

# Sidebar options
st.sidebar.header("Options")
mode = st.sidebar.selectbox("Mode", ["Chat", "Upload Syllabus (CSV)", "Quiz Generator", "About"])
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2)
max_tokens = st.sidebar.slider("Max tokens (response length)", 100, 1500, 700, step=50)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are Shiksha AI, a friendly tutor for Class 10 students. Explain simply in Bengali when possible and provide step-by-step solutions for math problems."}
    ]
if "syllabus_df" not in st.session_state:
    st.session_state.syllabus_df = None

def call_openai_chat(user_message: str, temp: float = 0.2, max_t: int = 700) -> str:
    """Call OpenAI ChatCompletion and return assistant text."""
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not set in .env"
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_message})
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
            temperature=float(temp),
            max_tokens=int(max_t)
        )
        assistant_msg = resp.choices[0].message["content"].strip()
    except Exception as e:
        assistant_msg = f"OpenAI error: {e}"
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg

def render_conversation():
    """Render chat conversation in the right column."""
    for m in st.session_state.messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            continue
        if role == "user":
            st.markdown(f"**You:** {content}")
        elif role == "assistant":
            st.markdown(f"**Shiksha AI:** {content}")

# Mode: Chat
if mode == "Chat":
    st.header("প্রশ্ন করো — Shiksha AI উত্তর দেবে")
    col1, col2 = st.columns([3,1])
    with col1:
        user_input = st.text_area("তোমার প্রশ্ন লিখো (বাংলা/ইংরেজি)", height=180, key="user_input")
        send = st.button("Send", key="send_button")
        if send and user_input.strip():
            with st.spinner("উত্তর তৈরি হচ্ছে..."):
                answer = call_openai_chat(user_input.strip(), temp=temperature, max_t=max_tokens)
                st.markdown("### উত্তর — Shiksha AI")
                st.write(answer)
                # clear input
                st.session_state["user_input"] = ""
    with col2:
        st.markdown("### Conversation")
        render_conversation()
        if st.button("Clear Conversation"):
            st.session_state.messages = [
                {"role": "system", "content": "You are Shiksha AI, a friendly tutor for Class 10 students. Explain simply in Bengali when possible and provide step-by-step solutions for math problems."}
            ]
            st.experimental_rerun()

# Mode: Upload Syllabus (CSV)
elif mode == "Upload Syllabus (CSV)":
    st.header("সিলেবাস / নোট আপলোড করো (CSV)")
    st.markdown("CSV ফরম্যাট: `question,answer` বা `topic,content`. উদাহরণ নিচে দেওয়া আছে।")
    uploaded = st.file_uploader("Upload CSV (class10_faq.csv)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.syllabus_df = df
            st.success("CSV আপলোড সফল 👍")
            st.dataframe(df.head(20))
        except Exception as e:
            st.error(f"CSV পড়তে সমস্যা: {e}")
    if st.session_state.syllabus_df is not None:
        query = st.text_input("সিলেবাস থেকে কী খুঁজবে? (search term)", key="search_term")
        if st.button("Search in syllabus"):
            df = st.session_state.syllabus_df
            if query.strip() == "":
                st.info("অনুগ্রহ করে একটি search term প্রদান করো।")
            else:
                mask = df.astype(str).apply(lambda r: r.str.contains(query, case=False, na=False).any(), axis=1)
                results = df[mask]
                if results.empty:
                    st.info("কোনো মিল পাওয়া যায়নি। তুমি চাইলে Shiksha AI-কে বিষয়টি ব্যাখ্যা করতে বলো।")
                else:
                    st.write(results)
                    if st.button("Explain selected results (with AI)"):
                        combined = "\n\n".join(results.astype(str).apply(lambda r: " | ".join(r.values), axis=1).tolist())
                        prompt = f"শ্রেণি ১০ ছাত্রদের জন্য সহজ বাংলা ভাষায় নিচের বিষয়টি ব্যাখ্যা করো:\n\n{combined}"
                        with st.spinner("ব্যাখ্যা তৈরি হচ্ছে..."):
                            ans = call_openai_chat(prompt, temp=temperature, max_t=max_tokens)
                            st.markdown("### ব্যাখ্যা")
                            st.write(ans)

# Mode: Quiz Generator
elif mode == "Quiz Generator":
    st.header("Quick MCQ Generator (Class 10)")
    topic = st.text_input("বিষয়/টপিক (যেমন: 'Quadratic Equations' বা 'Cell Structure')", key="quiz_topic")
    num_q = st.slider("Number of MCQs", 1, 10, 5, key="num_q")
    prefer_bengali = st.checkbox("উত্তর বাংলায় চাই", value=True)
    if st.button("Generate Quiz"):
        if topic.strip() == "":
            st.warning("অনুগ্রহ করে একটি টপিক লিখো।")
        else:
            lang_note = "বাংলা" if prefer_bengali else "English"
            prompt = (
                f"Generate {num_q} multiple choice questions for Class 10 students on the topic '{topic}'. "
                f"Provide each question, 4 options labelled A-D, and indicate the correct option letter. "
                f"Keep language simple ({lang_note}). Also include short explanation for the correct answer."
            )
            with st.spinner("Quiz তৈরি হচ্ছে..."):
                quiz_text = call_openai_chat(prompt, temp=temperature, max_t=max_tokens)
                st.markdown("### Generated Quiz")
                st.write(quiz_text)
                # also save quiz text to session for download
                st.session_state.latest_quiz = quiz_text
    if "latest_quiz" in st.session_state:
        st.download_button("Download Quiz as TXT", st.session_state.latest_quiz, file_name="quiz.txt")

# Mode: About
elif mode == "About":
    st.header("Shiksha AI — Project Info")
    st.markdown("""
- এটি একটি **MVP** — Class 10 ছাত্রদের প্রশ্নের উত্তর, সিলেবাস অনুসন্ধান এবং কুইজ জেনারেট করতে সক্ষম।
- চালাতে: `.env` ফাইলে `OPENAI_API_KEY=sk-...` রাখো, এবং `pip install -r requirements.txt` করে `streamlit run app.py` চালাও।
- যদি `app.py` ফাইলটি 0 byte হয়ে থাকে, এই কোডটি পেস্ট করে সেভ করে দাও।
- কোনো সমস্যা হলে Stack trace কপি করে পাঠাবে — আমি দেখব।
""")
    st.markdown("### Quick tips")
    st.markdown("""
- PowerShell-এ space ছিল এমন path-এ যেতে: `cd \"C:\\Folders\\Shiksha AI\"`
- ডেপ্লয় করতে চাইলে GitHub-এ পুশ করে Streamlit Community Cloud ব্যবহার করো।
""")

# Footer
st.markdown("---")
st.caption("Developed for Shiksha AI — Class 10. যদি চাও আমি sample syllabus CSV ও requirements.txt দিয়েও দিতে পারি।")

