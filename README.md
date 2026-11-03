# Brilliant-Directories-Knowledge-Brain-using-Hybrid-RAG

## 📌 Project Overview

Brilliant Directories Knowledge Brain is an AI-powered customer support assistant built using **Retrieval-Augmented Generation (RAG)**.

Brilliant Directories provides a platform for creating and managing online business directories and membership websites. It also maintains extensive support documentation covering features such as membership plans, subscriptions, member management, payments, and content publishing.

Finding specific solutions across multiple support articles can be time-consuming. This project addresses that challenge by providing an intelligent assistant that retrieves relevant information from support documentation and generates context-aware answers with source references.

## 🎯 Project Objectives

- Automate information retrieval from Brilliant Directories support documentation.
- Reduce manual searching across multiple support articles.
- Implement Hybrid Retrieval using semantic and keyword search.
- Generate context-aware answers using an LLM.
- Provide relevant source references for generated answers.

## 📊 Data Collection

- **Source:** Brilliant Directories Support Website.
- **Sitemap URL:** https://support.brilliantdirectories.com/support/sitemap.xml
- **Total URLs Collected:** 1,700
- **Support Article URLs:** 1,698
- **Successfully Extracted Articles:** 1,541
- **Data Type:** Unstructured Text Data.

### Web Scraping

- Extracted URLs from the XML sitemap.
- Used Python and BeautifulSoup for web scraping.
- Extracted article content from HTML pages using the `article#article-body` element.
- Collected article content and source URLs for further processing.

## ⚙️ Data Preprocessing

- Removed HTML tags and unwanted content.
- Cleaned and standardized extracted text.
- Split documents into smaller chunks.
- **Chunk Size:** 1,200 characters.
- **Chunk Overlap:** 150 characters.
- Added metadata, including source URLs, to document chunks.

## 🏗️ RAG Architecture

The project follows three major stages:

### 1. Indexing

- Processed and split support articles into chunks.
- Generated embeddings using HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`).
- Stored document chunks and embeddings in ChromaDB.

### 2. Retrieval – Hybrid RAG

Implemented an **Ensemble Retriever** combining two retrieval techniques:

- **Vector Search:** Retrieves semantically similar documents using embeddings and ChromaDB.
- **BM25 Search:** Retrieves relevant documents based on keyword matching.
- **Ensemble Retriever:** Combines both retrieval methods to improve relevant document retrieval.

### 3. Generation

- Passed the user query and retrieved context to the Groq-powered LLM.
- Generated context-aware answers based on retrieved documents.
- Included source references with the generated responses.
- Used a fallback response when sufficient information was not available in the retrieved documents.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Scraping | BeautifulSoup, Requests |
| Data Processing | Python |
| Embedding Model | HuggingFace Sentence Transformers |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Retrieval Strategy | Hybrid RAG / Ensemble Retriever |
| LLM | Groq |
| Framework | LangChain |
| User Interface | Streamlit |
| Evaluation | RAGAS |

## 📈 RAG Evaluation

Evaluated the RAG pipeline using the **RAGAS framework** with the following metrics:

1. **Context Precision** – Measures the relevance of retrieved context.
2. **Context Recall** – Measures how much relevant information is retrieved.
3. **Faithfulness** – Checks whether generated answers are supported by retrieved context.
4. **Answer Relevancy** – Measures how relevant the generated answer is to the user's question.
5. **Factual Correctness** – Evaluates the factual accuracy of generated answers against reference answers.

## 💻 Application Features

- Interactive Streamlit interface.
- Natural language question answering.
- Hybrid document retrieval.
- Context-aware answer generation.
- Source references for generated answers.
- Clear chat functionality.

## 🔄 Project Workflow

```text
Brilliant Directories Sitemap
          ↓
      URL Extraction
          ↓
       Web Scraping
          ↓
 BeautifulSoup Content Extraction
          ↓
    Data Preprocessing
          ↓
     Document Chunking
          ↓
    Embedding Generation
          ↓
       ChromaDB
          ↓
   Hybrid Retrieval
    ↙          ↘
Vector Search   BM25 Search
    ↘          ↙
  Ensemble Retriever
          ↓
     Retrieved Context
          ↓
      Groq LLM
          ↓
   Context-Aware Answer
          ↓
   Streamlit Application
          ↓
 Answer with Source References
```

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd Brilliant-Directories-Knowledge-Brain
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project directory:

```env
GROQ_API_KEY= groq_api_key
```

### 5. Run the Application

```bash
streamlit run app.py
```

## 🎯 Business Impact

- Reduces manual search effort.
- Provides faster access to support information.
- Improves documentation accessibility.
- Delivers context-aware answers with source references.
- Helps reduce repetitive customer support queries.

## 🔮 Future Enhancements

- Expand the knowledge base with additional documentation.
- Improve retrieval accuracy through reranking.
- Expand the evaluation dataset and monitor RAG performance.
- Add conversation memory for multi-turn interactions.

  
## 🎥 Project Demo

Watch the complete demonstration of Brilliant Directories Knowledge Brain:

[▶️ Watch Project Demo]("C:\Users\user\Desktop\Brillient directories RAG\Brillient directories Knowledge brain.mp4")

## 👨‍💻 Author

**Harish Alakuntla**

AI/ML | Data Science | Generative AI

LinkedIn: 
