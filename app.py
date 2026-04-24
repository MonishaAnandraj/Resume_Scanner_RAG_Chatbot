import streamlit as st
import os
import pandas as pd
from utils import process_pdfs, get_answer

st.set_page_config(page_title="Resume ATS System", layout="wide")

st.title("📄 Resume RAG Chatbot (ATS Ranking System)")


# -----------------------------
# UPLOAD RESUMES
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload multiple resumes (PDF)",
    type="pdf",
    accept_multiple_files=True
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# -----------------------------
# PROCESS RESUMES
# -----------------------------
if uploaded_files and st.button("Process Resumes"):
    with st.spinner("Processing resumes..."):
        os.makedirs("temp", exist_ok=True)

        file_paths = []
        for file in uploaded_files:
            path = os.path.join("temp", file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(path)

        st.session_state.vectorstore = process_pdfs(file_paths)

    st.success("✅ Resumes processed successfully!")


# -----------------------------
# QUERY SECTION
# -----------------------------
query = st.text_input("Ask something (e.g., Find Python developer)")

if query and st.session_state.vectorstore:
    with st.spinner("Ranking candidates..."):
        results = get_answer(st.session_state.vectorstore, query)

    if results:
        st.subheader("📌 Ranked Candidates")

        df = pd.DataFrame(results)

        df["skills"] = df["skills"].apply(lambda x: ", ".join(x))

        df = df[["name", "skills", "score", "source", "match"]]

        st.dataframe(df, use_container_width=True)

    else:
        st.warning("No matching resumes found.")