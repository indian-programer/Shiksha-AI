import streamlit as st
import os
import pandas as pd
import openai

st.set_page_config(page_title="Shiksha AI")
st.title("Shiksha AI — Debug")

# load api key from env / streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

st.write("OpenAI key loaded:", bool(openai.api_key))

# get key
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
elif os.getenv("OPENAI_API_KEY"):
    api_key = os.getenv("OPENAI_API_KEY")
else:
    api_key = st.text_input("Enter OpenAI API key (temporary)", type="password")

if not api_key:
    st.warning("Please set OPENAI_API_KEY in Streamlit Secrets or enter it here.")
    st.stop()

# create client
try:
    client = OpenAI(api_key=api_key)
    st.success("OpenAI client created (OK)")
except Exception as e:
    st.error(f"Failed to create OpenAI client: {e}")
    st.stop()

# small connection test button
if st.button("Run connection test"):
    try:
        r = client.models.list()
        st.write("Models count:", len(r.data) if hasattr(r, "data") else "unknown")
        st.write("Sample model:", r.data[0].id if hasattr(r, "data") and len(r.data)>0 else str(r)[:200])
    except Exception as e:
        st.error(f"Connection test failed: {e}")

# conversation state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content":"You are a helpful assistant. Prefer Bengali answers."}]

st.subheader("Conversation")
for m in st.session_state.messages:
    if m["role"]=="user":
        st.markdown(f"**You:** {m['content']}")
    elif m["role"]=="assistant":
        st.markdown(f"**Bot:** {m['content']}")

st.markdown("---")
user_input = st.text_input("Type your message and press Enter", key="input")

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.spinner("Thinking..."):
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages,
                max_tokens=400,
                temperature=0.3,
            )
            # parse response (v1.x)
            choice = resp.choices[0]
            assistant_text = choice.message.content.strip()
            st.session_state.messages.append({"role":"assistant","content":assistant_text})
            st.experimental_rerun()
        except Exception as e:
            st.error(f"API call failed: {e}")
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







