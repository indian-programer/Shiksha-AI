import streamlit as st
import os
import pandas as pd
import openai
from typing import Optional

# Page
st.set_page_config(page_title="Shiksha AI")
st.title("Shiksha AI — Debug")

# Load API key from environment / streamlit secrets
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
openai.api_key = OPENAI_KEY
st.write("OpenAI key loaded:", bool(OPENAI_KEY))

# -------------------------
# Helper: call OpenAI chat (robust)
# -------------------------
def call_openai_chat(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.2, max_tokens: int = 700) -> str:
    """Call OpenAI ChatCompletion and return assistant text or an error string."""
    if not openai.api_key:
        return "Error: OpenAI API key not set."

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # safe access to response text
        try:
            return resp.choices[0].message["content"].strip()
        except Exception:
            # fallback: stringify response
            return str(resp)[:1000]
    except Exception as e:
        return f"OpenAI call failed: {e}"

# -------------------------
# Session state for messages (Chat mode)
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content":"You are a helpful assistant. Prefer Bengali answers."}]

# Sidebar options
st.sidebar.header("Options")
mode = st.sidebar.selectbox("Mode", ["Chat", "Upload Syllabus (CSV)", "Quiz Generator", "About"])
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2, step=0.1)
max_tokens = st.sidebar.slider("Max tokens", 100, 1500, 700, step=50)

# -------------------------
# Mode: Chat
# -------------------------
if mode == "Chat":
    st.subheader("💬 Chat Mode")
    st.write("Ask questions in Bengali or English. The assistant will reply using OpenAI.")
    # Show conversation
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        elif m["role"] == "assistant":
            st.markdown(f"**Bot:** {m['content']}")

    # input and send
    user_text = st.text_input("Type your message and press Enter", key="chat_input")
    if st.button("Send", key="chat_send") or (user_text and st.session_state.get("last_input") != user_text):
        if user_text and user_text.strip():
            st.session_state.messages.append({"role":"user","content":user_text.strip()})
            with st.spinner("Thinking..."):
                prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                reply = call_openai_chat(user_text, temperature=temperature, max_tokens=max_tokens)
                st.session_state.messages.append({"role":"assistant","content":reply})
            st.experimental_rerun()
        else:
            st.warning("প্রশ্ন লিখুন!")

# -------------------------
# Mode: Upload Syllabus (CSV)
# -------------------------
elif mode == "Upload Syllabus (CSV)":
    st.subheader("📄 Upload syllabus (CSV) — Searchable")
    uploaded = st.file_uploader("Upload syllabus CSV", type=["csv"], key="uploader")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:", df.head(20))
        except Exception as e:
            st.error(f"CSV পড়তে সমস্যা: {e}")
            df = pd.DataFrame()
        if not df.empty:
            keyword = st.text_input("Search keyword:", key="search_keyword")
            if keyword and keyword.strip():
                mask = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)
                results = df[mask]
                st.write(results)
                if st.button("Explain selected results (with AI)"):
                    combined = "\n\n".join(results.astype(str).apply(lambda r: " | ".join(r.values.astype(str)), axis=1).tolist())
                    prompt = f"ছাত্রদের জন্য সহজ বাংলায় নিচের বিষয়টি ব্যাখ্যা করো:\n\n{combined}"
                    with st.spinner("ব্যাখ্যা তৈরি হচ্ছে..."):
                        ans = call_openai_chat(prompt, temperature=temperature, max_tokens=max_tokens)
                        st.markdown("### ব্যাখ্যা")
                        st.write(ans)
        else:
            st.info("ফাইল পড়া যায়নি বা ফাইল খালি।")

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
- On Streamlit Cloud: go to App → Settings → Secrets and add `OPENAI_API_KEY = "sk-..."`.
- Locally: create `~/.streamlit/secrets.toml` with:

"""
    )

# Footer
st.markdown("---")
st.caption("Developed for Shiksha AI — provide a sample syllabus CSV & requirements.txt if you want further help.")
