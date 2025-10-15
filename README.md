.

🧠 Knowledge-Base Search Engine with RAG
An intelligent search engine that ingests multiple PDF documents and uses a Retrieval-Augmented Generation (RAG) pipeline to provide synthesized, context-aware answers to user questions. Instead of just finding keywords, this application understands the content of your documents and generates human-like responses.

This project allows you to create your own private, powerful search engine for any collection of documents.

📸 Demo
(Recommended) A short GIF is the best way to show how the application works. Record a quick demo of you uploading a PDF, running the app, and asking a question.

✨ Core Features
Document Ingestion: Seamlessly process multiple PDF files at once. The system automatically extracts text, splits it into manageable chunks, and prepares it for searching.

Vector Embeddings: Uses state-of-the-art sentence-transformer models to convert text chunks into numerical vectors, capturing the semantic meaning of the content.

Efficient Vector Storage: Leverages ChromaDB to store and efficiently search through thousands of text embeddings, ensuring fast retrieval of relevant information.

Retrieval-Augmented Generation (RAG): Implements a powerful RAG pipeline that:

Retrieves the most relevant document chunks for a given question.

Augments a Large Language Model (LLM) with this specific context.

Generates a coherent, accurate answer based only on the provided documents.

Interactive Web Interface: A clean and simple UI built with Flask and Jinja2 allows users to easily submit questions and view the generated answers and their sources.

⚙️ How It Works: The RAG Pipeline
The application follows a two-stage process:

1. Ingestion (ingest.py)
This is a one-time process you run whenever you add or update your documents.

Load: All PDF files from the /documents folder are loaded.

Split: The text from each document is split into smaller, overlapping chunks. This helps maintain context without overwhelming the language model.

Embed: Each chunk is passed through a Hugging Face embedding model (all-MiniLM-L6-v2) to create a vector.

Store: These vectors and their corresponding text are stored in a local ChromaDB vector database in the /db folder.

2. Question-Answering (app.py)
This process runs in real-time whenever a user asks a question.

User Query: A user submits a question through the web interface.

Embed Query: The user's question is converted into a vector using the same embedding model.

Search & Retrieve: The application searches the ChromaDB database to find the text chunks whose vectors are most similar to the question's vector.

Augment & Generate: The retrieved text chunks (the context) and the user's question are inserted into a prompt. This prompt is then sent to a Large Language Model (e.g., google/flan-t5-base) which generates a final answer based on the provided context.

🛠️ Tech Stack
Backend Framework: Flask

LLM Orchestration: LangChain

LLM & Embeddings: Hugging Face Hub (Transformers & Sentence-Transformers)

Vector Database: ChromaDB

PDF Processing: PyPDF

Frontend: HTML & Jinja2

📂 Project Structure
knowledge-base-search-engine/
├── app.py             # Main Flask application with API endpoints
├── ingest.py          # Script to process and embed documents
├── requirements.txt   # List of all Python dependencies
├── README.md          # You are here!
├── documents/         # --- Place your PDF files in this folder ---
└── templates/
    └── index.html     # The HTML file for the web interface
🚀 Getting Started: Setup and Installation
Follow these steps to get the project running on your local machine.

1. Prerequisites
Python 3.9 or higher

Git for cloning the repository

2. Clone the Repository
Bash

git clone <your-repository-link>
cd knowledge-base-search-engine
3. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

Bash

# Create the virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
4. Install Dependencies
Install all the required libraries from the requirements.txt file.

Bash

pip install -r requirements.txt
5. Get Your Hugging Face API Token
The application uses a model from the Hugging Face Hub, which requires a free API token.

Create an account on huggingface.co.

Navigate to Settings -> Access Tokens.

Generate a new token with the read role.

Important: The provided app.py code requires you to hardcode this token. Open the app.py file and paste your token where indicated.

▶️ How to Run the Application
Step 1: Add Your Documents
Place all the PDF files you want to search through into the /documents folder. You can start by saving a web page (like a Wikipedia article) as a PDF.

Step 2: Run the Ingestion Script
This step "teaches" the search engine about your documents. Run this script once, or whenever you add/change the files in the /documents folder.

Bash

python ingest.py
This will create a /db folder containing the vector database.

Step 3: Run the Web Application
Start the Flask server.

Bash

python app.py
Step 4: Open Your Search Engine!
Open your web browser and navigate to the URL shown in the terminal, which is usually:

http://127.0.0.1:5001

You can now ask questions about the content of your PDFs!

🔧 Customization
You can easily customize the application by modifying app.py:

Change the LLM: To use a different model, simply change the LLM_REPO_ID variable to another model from the Hugging Face Hub (e.g., mistralai/Mistral-7B-Instruct-v0.2).

Customize the Prompt: Modify the prompt_template string to change how the AI is instructed to answer questions. You can alter its tone, format, or constraints.

Adjust Retrieval: Change the k value in the retriever definition to fetch more or fewer document chunks for context.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.








