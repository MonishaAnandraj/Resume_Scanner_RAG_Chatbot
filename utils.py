import re
import spacy
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, util

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# ROLE → SKILL MAP (ATS INTELLIGENCE)
# -----------------------------
QUERY_SKILL_MAP = {
    "python developer": ["python", "flask", "django", "fastapi", "api", "sql"],
    "data analyst": ["sql", "excel", "python", "powerbi", "tableau"],
    "data scientist": ["python", "machine learning", "deep learning", "statistics"],
    "frontend developer": ["react", "html", "css", "javascript"],
    "backend developer": ["node", "java", "sql", "api"]
}


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
# NAME EXTRACTION (FIXED)
# -----------------------------
def extract_name(text):
    lines = text.split("\n")

    ignore_keywords = [
        "resume", "cv", "profile", "education",
        "experience", "skills", "technical",
        "data science", "trainee", "engineer",
        "developer", "intern"
    ]

    for line in lines[:25]:
        clean = line.strip()

        if len(clean) < 3:
            continue

        if any(word in clean.lower() for word in ignore_keywords):
            continue

        # must be real name pattern (2-3 words, capitalized)
        if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+){1,2}$", clean):
            return clean

    return "Unknown Candidate"

# -----------------------------
# SKILL EXTRACTION
# -----------------------------
def extract_skills(text):
    skills_db = [
        "python", "java", "react", "node", "sql",
        "machine learning", "deep learning", "ai",
        "flask", "django", "aws", "azure",
        "powerbi", "tableau", "excel"
    ]

    text_lower = text.lower()
    return list(set([s for s in skills_db if s in text_lower]))


# -----------------------------
# SCORE (SEMANTIC + ROLE INTELLIGENCE)
# -----------------------------
def calculate_score(text, query):

    query = query.lower()

    expanded = QUERY_SKILL_MAP.get(query, [query])
    query_text = " ".join(expanded)

    semantic_score = util.cos_sim(
        model.encode(text),
        model.encode(query_text)
    ).item()

    skill_hits = sum(1 for s in expanded if s in text.lower())

    final_score = (semantic_score * 100) + (skill_hits * 5)

    return round(final_score, 2)


# -----------------------------
# PROCESS PDFS
# -----------------------------
def process_pdfs(pdf_files):
    documents = []

    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        pages = loader.load()

        full_text = ""

        for page in pages:
            full_text += page.page_content + " "

        full_text = clean_text(full_text)

        # IMPORTANT: treat each resume as ONE document
        documents.append({
            "page_content": full_text,
            "metadata": {"source": pdf}
        })

    # Convert to LangChain documents manually
    from langchain_core.documents import Document

    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in documents
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# -----------------------------
# GET ANSWER (ATS LOGIC FIXED)
# -----------------------------
def get_answer(vectorstore, query):

    docs = vectorstore.similarity_search(query, k=5)

    results = []

    for doc in docs:
        full_text = doc.page_content
        source = doc.metadata.get("source", "unknown")

        name = extract_name(full_text)
        skills = extract_skills(full_text)
        score = calculate_score(full_text, query)

        results.append({
            "name": name,
            "skills": skills,
            "score": score,
            "source": source,
            "match": full_text[:300]
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)