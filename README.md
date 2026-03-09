# AI Knowledge Retrieval System
Live at : https://ai-knowledge-retrieval-system.streamlit.app

## 📌 Project Overview

The **AI Knowledge Retrieval System** is an intelligent application that allows users to upload documents and retrieve relevant information by asking natural language questions. The system processes documents, converts them into vector embeddings, stores them in a vector database, and retrieves the most relevant content using semantic search.

This project demonstrates how modern **AI-powered information retrieval systems** work using embeddings, vector search, and ranking techniques.

---

## 🚀 Features

* Upload and process PDF documents
* Intelligent document chunking
* Generate vector embeddings
* Fast semantic search using vector similarity
* Re-ranking of results for improved accuracy
* Question-based information retrieval
* Interactive UI using Streamlit

---

## 🧠 System Architecture

User Query
↓
Query Processing
↓
Embedding Generation
↓
Vector Similarity Search (FAISS / Vector Store)
↓
Re-ranking of Results
↓
Return Most Relevant Answer

---

## 🛠 Tech Stack

**Frontend**

* Streamlit

**Backend**

* Python

**AI / NLP**

* Sentence Transformers
* Embedding Models

**Vector Database**

* FAISS

**Data Processing**

* PyPDF
* NumPy
* Pandas

---

## 📂 Project Structure

```
AI_Knowledge_Retrieval_System
│
├── app/
├── core/
├── loaders/
├── processing/
├── embeddings/
├── vector_store/
├── ranking/
├── services/
├── models/
├── evaluation/
│
├── main.py
├── demo.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```
git clone https://github.com/Swapnil454/ai-knowledge-retrieval-system.git
```

### 2️⃣ Navigate to Project

```
cd ai-knowledge-retrieval-system
```

### 3️⃣ Create Virtual Environment

```
python -m venv .venv
```

### 4️⃣ Activate Environment

Windows:

```
.venv\Scripts\activate
```

Linux / Mac:

```
source .venv/bin/activate
```

### 5️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run main.py
```

The application will start at:

```
http://localhost:8501
```

---

## 📊 Use Case

This system can be used for:

* Document search systems
* Knowledge base assistants
* Research document analysis
* Enterprise document retrieval
* AI-powered FAQ systems

---

## 🔮 Future Improvements

* Support multiple document formats
* Integrate LLM for better answer generation
* Improve ranking algorithms
* Add user authentication
* Deploy using Docker and cloud services

---

## 👨‍💻 Author

**Swapnil Shelke**

Aspiring Full Stack & AI Developer

GitHub:
https://github.com/Swapnil454

---

## ⭐ Support

If you like this project, please give it a **star on GitHub** ⭐
