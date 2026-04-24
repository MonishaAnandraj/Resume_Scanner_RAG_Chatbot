from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import spacy
import re
from sentence_transformers import SentenceTransformer, util

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# QUERY EXPANSION (IMPORTANT)
# -----------------------------
QUERY_SKILL_MAP = {
    "data analyst": ["sql", "excel", "python", "powerbi", "tableau", "statistics"],
    "data science": ["python", "machine learning", "sql", "statistics"],
    "backend developer": ["node", "java", "sql", "api"],
    "frontend developer": ["react", "html", "css", "javascript"],
    "python developer": ["python", "django", "flask", "api"]
}


# -----------------------------
# NAME EXTRACTION (FIXED)
# -----------------------------
def extract_name(text):
    lines = text.split("\n")

    ignore_words = ["resume", "cv", "profile", "email", "phone", "linkedin"]

    for line in lines[:20]:
        clean = line.strip()

        if len(clean) < 3:
            continue

        if any(word in clean.lower() for word in ignore_words):
            continue

        if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+$", clean):
            return clean

    return "Unknown Candidate"


# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    bad_words = ["cgpa", "bachelor", "master", "education"]

    lines = text.split("\n")
    filtered = []

    for line in lines:
        if len(line.strip()) < 3:
            continue

        if any(word in line.lower() for word in bad_words):
            continue

        filtered.append(line)

    return "\n".join(filtered)


# -----------------------------
# SKILL EXTRACTION (NLP)
# -----------------------------
def extract_skills_nlp(text):
    skills_db = [
        "python", "java", "react", "node", "sql",
        "machine learning", "deep learning", "ai",
        "flask", "django", "aws", "azure",
        "powerbi", "tableau", "excel"
    ]

    text_lower = text.lower()
    return list(set([s for s in skills_db if s in text_lower]))


# -----------------------------
# SEMANTIC + QUERY AWARE SCORE
# -----------------------------
def calculate_score(text, query):

    query_lower = query.lower()

    expanded = QUERY_SKILL_MAP.get(query_lower, [query_lower])
    expanded_query = " ".join(expanded)

    semantic_score = util.cos_sim(
        model.encode(text),
        model.encode(expanded_query)
    ).item()

    return round(semantic_score * 100, 2)


# -----------------------------
# PROCESS PDFs
# -----------------------------
def process_pdfs(pdf_files):
    documents = []

    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf
            doc.page_content = clean_text(doc.page_content)

        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# -----------------------------
# GET ANSWER (ATS LOGIC)
# -----------------------------
def get_answer(vectorstore, query):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(query)

    if not docs:
        return []

    resume_map = {}

    for doc in docs:
        text = doc.page_content
        source = doc.metadata.get("source", "unknown")

        if source not in resume_map:
            resume_map[source] = {"text": ""}

        resume_map[source]["text"] += " " + text

    results = []

    for source, data in resume_map.items():
        full_text = data["text"]

        name = extract_name(full_text)
        skills = extract_skills_nlp(full_text)

        score = calculate_score(full_text, query)

        results.append({
            "name": name,
            "skills": skills,
            "score": score,
            "source": source,
            "match": full_text[:300]
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)