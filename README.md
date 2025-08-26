Enhanced RAG Chatbot with arXiv Integration

An AI-powered chatbot built with Streamlit, LangChain, Together AI, HuggingFace embeddings, and Chroma, that can:
-> Answer questions from uploaded documents (RAG)
-> Search & summarize research papers from arXiv
-> Support multi-language interaction
-> Maintain conversational memory

Features
Research Paper Search (arXiv)
Automatically detects queries asking for "research papers".
Fetches metadata (title, authors, abstract, categories, publication date, link).
Summarizes top 3 abstracts using a Together AI LLM.

Document-based Q&A (RAG)
Upload PDF, TXT, DOCX, CSV files.
Documents are:
Split into chunks
Embedded using HuggingFace (thenlper/gte-large)
Stored in a Chroma vector DB
Supports Vectorstore retriever or Contextual compression retriever.
Uses ConversationalRetrievalChain with memory for contextual chat.

LLM Integration (Together AI)
Custom LLM wrapper (TogetherLLM) with adjustable params.
Supported models:
Llama-3.3-70B-Instruct-Turbo-Free (default)
Meta-Llama-3.1-70B-Instruct-Turbo (128K)
DeepSeek-V3
DeepSeek-R1-Distill-Llama-70B
Mixtral-8x7B
Configurable temperature & max tokens.

Chat Interface
Persistent chat history (st.session_state).
Multilingual assistant greetings (EN, FR, ES, ZH, etc.).
Clear chat option.
Sidebar with:
Example prompts
Retriever options
Model controls

Tech Stack
Frontend/UI → Streamlit
LLM API → Together AI
Document Embeddings → HuggingFace Embeddings
Vector DB → Chroma

LangChain → Retrieval, Memory, Conversational Chains

arXiv API → Research paper search
