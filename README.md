# PDF RAG Chatbot

A modern Streamlit app that lets you chat with your own PDF documents using Retrieval-Augmented Generation (RAG) and OpenAI's GPT models. Upload PDFs, ask questions, and get answers with references to the exact document, page, and chunk.

## Features
- 📄 Upload up to 2 PDF files and process them for Q&A
- 💬 Chat with your PDFs using natural language
- 🔍 Answers are referenced with source file, page, and chunk
- 🧠 Uses OpenAI GPT-4o and sentence-transformers for semantic search
- 🖥️ Simple, modern Streamlit UI

## Demo
![PDF RAG Chatbot Screenshot](https://github.com/antonyga/ragxpert/blob/main/ragxpert_screenshot.png)

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run pdf_chatbot.py
```

### 4. Configure OpenAI API Key
- Get your API key from [OpenAI API Keys](https://platform.openai.com/api-keys)
- Enter it in the sidebar of the app

### 5. Upload PDFs and Chat
- Upload up to 2 PDF files
- Ask questions about their content
- Answers will include references to the source file, page, and chunk

## How it Works
- PDFs are split into chunks and embedded using `sentence-transformers`
- Chunks are stored in a local ChromaDB vector database
- User questions are embedded and matched to relevant chunks
- OpenAI GPT-4o generates answers using the most relevant chunks as context
- Each answer cites the document, page, and chunk for transparency

## Deployment
You can deploy this app for free on [Streamlit Community Cloud](https://streamlit.io/cloud) or any Python app hosting platform.

## License
MIT License

---

> Built with ❤️ using Streamlit, OpenAI, and ChromaDB.
