# 📄 Resume RAG Chatbot (AI ATS Ranking System)

## 🚀 Overview

The **Resume RAG Chatbot** is an AI-powered resume screening system that uses **Retrieval-Augmented Generation (RAG)**, **semantic search**, and **NLP techniques** to rank candidates based on job queries like *Data Analyst, Python Developer, Backend Developer*, etc.

It acts like a mini **ATS (Applicant Tracking System)** that helps recruiters automatically find the best matching candidates from multiple resumes.

---

## ✨ Features

* 📂 Upload multiple PDF resumes
* 🔍 AI-powered semantic resume search (FAISS + embeddings)
* 📊 Intelligent candidate ranking system (% score)
* 🧠 NLP-based skill extraction (spaCy + skill dictionary)
* 🎯 Job role understanding (Data Analyst → SQL, Excel, PowerBI, etc.)
* 📑 Clean Streamlit UI with tabular results
* ⚡ Fast vector-based retrieval system

---

## 🧠 How It Works

1. **Upload Resumes (PDF)**
2. Text is extracted and cleaned
3. Documents are split into chunks
4. Embeddings are created using HuggingFace models
5. FAISS stores vector embeddings
6. User enters a job query
7. System:

   * Expands query into relevant skills
   * Computes semantic similarity score
   * Extracts candidate skills using NLP
   * Ranks candidates based on relevance
8. Results displayed in ranked table format

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit 🎨
* LangChain 🧠
* FAISS (Vector Database)
* HuggingFace Sentence Transformers
* spaCy (NLP)
* PyPDF

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/resume-rag-chatbot.git
cd resume-rag-chatbot
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Install spaCy model

```bash
python -m spacy download en_core_web_sm
```

---

### 4. Run Application

```bash
streamlit run app.py
```

---

## 🧪 Sample Query

```
Find Data Analyst
```

### Output:

* Ranked candidates
* Skill match (SQL, Python, Excel, PowerBI)
* Semantic score (%)
* Resume preview snippet
