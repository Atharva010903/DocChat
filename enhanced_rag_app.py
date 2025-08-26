####################################################################
#                         import
####################################################################

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import re

# Import Together AI as the main LLM service
from together import Together

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import format_document

# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)

# text_splitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# HuggingFace for embeddings (free)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Import streamlit
import streamlit as st

# Custom LLM wrapper for Together AI
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun

####################################################################
#                    Together AI LLM Wrapper
####################################################################

class TogetherLLM(LLM):
    """Custom LLM wrapper for Together AI"""
    
    api_key: str
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    @property
    def _llm_type(self) -> str:
        return "together"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            client = Together(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

####################################################################
#              Config: LLM services, assistant language,...
####################################################################

# Together AI Configuration
API_KEY = "tgp_v1_auHD1U7Mvm_VSPvfSZoVQ9m_woHoVtyU06DJ60ln-R0"

AVAILABLE_MODELS = {
    "Llama 3.3 70B (Default)": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "Llama 3.1 70B (Newer, 128K)": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "DeepSeek V3 (General, 128K)": "deepseek-ai/DeepSeek-V3",
    "DeepSeek R1 Distill Llama 70B (Reasoning, 8K)": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Mixtral 8x7B Instruct (8K)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

MODEL_CONTEXT_WINDOWS = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": 8192,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": 128000,
    "deepseek-ai/DeepSeek-V3": 128000,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 8192,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 8192,
}

dict_welcome_message = {
    "english": "How can I assist you today? I can help with your documents and also search for research papers on arXiv!",
    "french": "Comment puis-je vous aider aujourd'hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Contextual compression",
    "Vectorstore backed retriever",
]

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)

####################################################################
#                    arXiv Integration
####################################################################

class ArxivSearcher:
    """Class to handle arXiv paper searches"""
    
    @staticmethod
    def search_arxiv_papers(query: str, max_results: int = 5) -> List[dict]:
        """
        Search arXiv for papers related to the query
        Returns a list of paper dictionaries with metadata
        """
        try:
            # Clean and format the query
            search_query = query.replace(" ", "+")
            url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
            
            response = requests.get(url)
            if response.status_code != 200:
                return []
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = {}
                
                # Extract title
                title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                paper['title'] = title_elem.text.strip() if title_elem is not None else "No title"
                
                # Extract authors
                authors = []
                for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                    name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                    if name_elem is not None:
                        authors.append(name_elem.text)
                paper['authors'] = authors
                
                # Extract abstract
                summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                paper['abstract'] = summary_elem.text.strip() if summary_elem is not None else "No abstract"
                
                # Extract arXiv ID and URL
                id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                if id_elem is not None:
                    paper['url'] = id_elem.text
                    paper['arxiv_id'] = id_elem.text.split('/')[-1]
                
                # Extract published date
                published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                if published_elem is not None:
                    paper['published'] = published_elem.text[:10]  # Just the date part
                
                # Extract categories
                categories = []
                for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                paper['categories'] = categories
                
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            st.error(f"Error searching arXiv: {str(e)}")
            return []
    
    @staticmethod
    def format_papers_for_display(papers: List[dict]) -> str:
        """Format papers for display in Streamlit"""
        if not papers:
            return "No papers found."
        
        formatted = "## 📚 Research Papers from arXiv\n\n"
        for i, paper in enumerate(papers, 1):
            formatted += f"### {i}. {paper['title']}\n\n"
            formatted += f"**Authors:** {', '.join(paper['authors'])}\n\n"
            formatted += f"**Published:** {paper.get('published', 'N/A')}\n\n"
            formatted += f"**Categories:** {', '.join(paper['categories'])}\n\n"
            formatted += f"**Abstract:** {paper['abstract'][:300]}...\n\n"
            formatted += f"**arXiv URL:** {paper['url']}\n\n"
            formatted += "---\n\n"
        
        return formatted

####################################################################
#            Create app interface with streamlit
####################################################################
st.set_page_config(page_title="Enhanced RAG Chat With Your Data & Research Papers")

st.title("🤖 Enhanced RAG Chatbot with arXiv Integration")

# Initialize Together AI client
if not API_KEY:
    st.error("Error: Together AI API key is missing. Please replace the placeholder with your actual key.")
    st.stop()

def expander_model_parameters():
    """Add model selection and parameters for Together AI."""
    st.session_state.LLM_provider = "Together AI"

    with st.expander("**Models and parameters**"):
        # Model selection
        model_display_names = list(AVAILABLE_MODELS.keys())
        selected_display_name = st.selectbox(
            "Choose Together AI model", 
            model_display_names,
            index=0  # Default to first model
        )
        st.session_state.selected_model = AVAILABLE_MODELS[selected_display_name]
        
        # Display context window info
        context_window = MODEL_CONTEXT_WINDOWS.get(st.session_state.selected_model, 8192)
        st.info(f"Context window for this model: {context_window:,} tokens")

        # Model parameters
        st.session_state.temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        st.session_state.max_tokens = st.slider(
            "max_tokens",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
        )

def sidebar_and_documentChooser():
    """Create the sidebar and the a tabbed pane for document/vectorstore management."""

    with st.sidebar:
        st.caption(
            "🚀 A retrieval augmented generation chatbot with arXiv research paper integration, powered by 🔗 Langchain and Together AI"
        )
        st.write("")

        st.write("**🤖 Together AI Models (Free Access)**")
        expander_model_parameters()

        st.divider()
        
        # Assistant language
        st.session_state.assistant_language = st.selectbox(
            f"Assistant language", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retrievers")
        
        st.session_state.retriever_type = st.selectbox(
            f"Select retriever type", list_retriever_types
        )

        st.divider()
        st.subheader("🔬 Research Features")
        st.session_state.enable_arxiv = st.checkbox(
            "Enable arXiv paper search", 
            value=True, 
            help="When enabled, you can ask me to search for research papers"
        )
        
        if st.session_state.enable_arxiv:
            st.session_state.max_papers = st.slider(
                "Max papers to retrieve",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
            )

    # Tabbed Pane: Create a new Vectorstore | Open a saved Vectorstore
    tab_new_vectorstore, tab_open_vectorstore = st.tabs(
        ["Create a new Vectorstore", "Open a saved Vectorstore"]
    )
    
    with tab_new_vectorstore:
        # 1. Select documents
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Select documents**",
            accept_multiple_files=True,
            type=(["pdf", "txt", "docx", "csv"]),
        )
        
        # 2. Process documents
        st.session_state.vector_store_name = st.text_input(
            label="**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). Please provide a valid dB name.**",
            placeholder="Vectorstore name",
        )
        
        # 3. Add a button to process documents and create a Chroma vectorstore
        st.button("Create Vectorstore", on_click=chain_RAG_blocks)
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except:
            pass

    with tab_open_vectorstore:
        # Open a saved Vectorstore
        st.write("Please select a Vectorstore:")
        import tkinter as tk
        from tkinter import filedialog

        clicked = st.button("Vectorstore chooser")
        if clicked:
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes("-topmost", 1)
            
            selected_vectorstore_path = filedialog.askdirectory(master=root)
            
            if selected_vectorstore_path == "":
                st.info("Please select a valid path.")
            else:
                with st.spinner("Loading vectorstore..."):
                    try:
                        # Load Chroma vectorstore
                        embeddings = select_embeddings_model()
                        st.session_state.vector_store = Chroma(
                            embedding_function=embeddings,
                            persist_directory=selected_vectorstore_path,
                        )
                        
                        # Create retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                        )
                        
                        # Create memory and ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )
                        
                        # Clear chat_history
                        clear_chat_history()
                        
                        st.session_state.selected_vectorstore_name = (
                            selected_vectorstore_path.split("/")[-1]
                        )
                        st.info(
                            f"**{st.session_state.selected_vectorstore_name}** is loaded successfully."
                        )
                        
                    except Exception as e:
                        st.error(e)

####################################################################
#        Process documents and create vectorstore (Chroma dB)
####################################################################

def delete_temp_files():
    """delete files from the './data/tmp' folder"""
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass

def langchain_document_loader():
    """Create document loaders for PDF, TXT, DOCX and CSV files."""
    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
        loader_kwargs={"encoding":"utf8"}
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents

def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def select_embeddings_model():
    """Select embeddings model - using HuggingFace free embeddings."""
    # Using a free HuggingFace embedding model
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        model_name="thenlper/gte-large",
        # You can use HF_TOKEN environment variable or leave empty for public models
    )
    return embeddings

def create_retriever(
    vector_store,
    embeddings,
    retriever_type="Contextual compression",
    base_retriever_search_type="similarity",
    base_retriever_k=16,
    compression_retriever_k=20,
):
    """Create a retriever."""
    
    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever
    elif retriever_type == "Contextual compression":
        compression_retriever = create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            k=compression_retriever_k,
        )
        return compression_retriever

def Vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    """create a vectorstore-backed retriever"""
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever

def create_compression_retriever(
    embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None
):
    """Build a ContextualCompressionRetriever."""
    
    # 1. splitting docs into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separator=". "
    )

    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
    )

    # 4. Reorder the documents
    reordering = LongContextReorder()

    # 5. create compressor pipeline and retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )

    return compression_retriever

def chain_RAG_blocks():
    """The RAG system is composed of: Retrieval, Memory, and Conversational Retrieval chain."""
    with st.spinner("Creating vectorstore..."):
        # Check inputs
        error_messages = []
        
        if not st.session_state.uploaded_file_list:
            error_messages.append("select documents to upload")
        if st.session_state.vector_store_name == "":
            error_messages.append("provide a Vectorstore name")

        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Please "
                + ", ".join(error_messages[:-1])
                + ", and "
                + error_messages[-1]
                + "."
            )
        else:
            st.session_state.error_message = ""
            try:
                # 1. Delete old temp files
                delete_temp_files()

                # 2. Upload selected documents to temp directory
                if st.session_state.uploaded_file_list is not None:
                    for uploaded_file in st.session_state.uploaded_file_list:
                        error_message = ""
                        try:
                            temp_file_path = os.path.join(
                                TMP_DIR.as_posix(), uploaded_file.name
                            )
                            with open(temp_file_path, "wb") as temp_file:
                                temp_file.write(uploaded_file.read())
                        except Exception as e:
                            error_message += str(e)
                    if error_message != "":
                        st.warning(f"Errors: {error_message}")

                    # 3. Load documents with Langchain loaders
                    documents = langchain_document_loader()

                    # 4. Split documents to chunks
                    chunks = split_documents_to_chunks(documents)
                    
                    # 5. Embeddings
                    embeddings = select_embeddings_model()

                    # 6. Create a vectorstore
                    persist_directory = (
                        LOCAL_VECTOR_STORE_DIR.as_posix()
                        + "/"
                        + st.session_state.vector_store_name
                    )

                    try:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_directory,
                        )
                        st.info(
                            f"Vectorstore **{st.session_state.vector_store_name}** is created successfully."
                        )

                        # 7. Create retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                        )

                        # 8. Create memory and ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )

                        # 9. Clear chat_history
                        clear_chat_history()

                    except Exception as e:
                        st.error(e)

            except Exception as error:
                st.error(f"An error occurred: {error}")

####################################################################
#                       Create memory
####################################################################

def create_memory():
    """Creates a ConversationBufferMemory for Together AI models"""
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
        input_key="question",
    )
    return memory

####################################################################
#          Create ConversationalRetrievalChain with memory
####################################################################

def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context to the LLM."""

    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template

def create_ConversationalRetrievalChain(
    retriever,
    chain_type="stuff",
    language="english",
):
    """Create a ConversationalRetrievalChain using Together AI."""

    # 1. Define the standalone_question prompt
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""",
    )

    # 2. Define the answer_prompt
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Add ConversationBufferMemory
    memory = create_memory()

    # 4. Instantiate Together AI LLMs
    standalone_query_generation_llm = TogetherLLM(
        api_key=API_KEY,
        model_name=st.session_state.selected_model,
        temperature=0.1,
        max_tokens=st.session_state.max_tokens,
    )
    
    response_generation_llm = TogetherLLM(
        api_key=API_KEY,
        model_name=st.session_state.selected_model,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
    )

    # 5. Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory

def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass

def detect_research_request(prompt: str) -> bool:
    """Detect if the user is asking for research papers."""
    research_keywords = [
        "research papers", "research paper", "papers", "paper",
        "arxiv", "arXiv", "academic", "study", "studies", 
        "publication", "publications", "literature", "scholarly",
        "look up", "search for", "find papers", "recent research"
    ]
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in research_keywords)

def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        # Check if user is asking for research papers
        if (hasattr(st.session_state, 'enable_arxiv') and 
            st.session_state.enable_arxiv and 
            detect_research_request(prompt)):
            
            # Search for research papers
            with st.spinner("🔍 Searching arXiv for research papers..."):
                arxiv_searcher = ArxivSearcher()
                papers = arxiv_searcher.search_arxiv_papers(
                    prompt, 
                    max_results=st.session_state.get('max_papers', 5)
                )
                
                if papers:
                    # Display papers
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    papers_content = arxiv_searcher.format_papers_for_display(papers)
                    
                    # Also get LLM response about the papers
                    paper_summaries = []
                    for paper in papers:
                        paper_summaries.append(f"Title: {paper['title']}\nAbstract: {paper['abstract']}")
                    
                    context_prompt = f"Based on these research papers, answer the user's question: {prompt}\n\nPapers:\n" + "\n\n".join(paper_summaries[:3])  # Limit to avoid token limits
                    
                    try:
                        llm = TogetherLLM(
                            api_key=API_KEY,
                            model_name=st.session_state.selected_model,
                            temperature=st.session_state.temperature,
                            max_tokens=st.session_state.max_tokens,
                        )
                        llm_response = llm._call(context_prompt)
                        
                        combined_response = f"{llm_response}\n\n{papers_content}"
                    except:
                        combined_response = papers_content
                    
                    st.session_state.messages.append({"role": "assistant", "content": combined_response})
                    
                    st.chat_message("user").write(prompt)
                    with st.chat_message("assistant"):
                        st.markdown(combined_response)
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    no_papers_msg = "I couldn't find any research papers for your query. Let me try to answer based on my knowledge or your uploaded documents."
                    st.session_state.messages.append({"role": "assistant", "content": no_papers_msg})
                    st.chat_message("user").write(prompt)
                    st.chat_message("assistant").write(no_papers_msg)
                    
                    # Fall back to regular RAG if available
                    if hasattr(st.session_state, 'chain'):
                        response = st.session_state.chain.invoke({"question": prompt})
                        answer = response["answer"]
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                            
                            # Display source documents if available
                            if "source_documents" in response and response["source_documents"]:
                                with st.expander("**Source documents**"):
                                    documents_content = ""
                                    for document in response["source_documents"]:
                                        try:
                                            page = " (Page: " + str(document.metadata["page"]) + ")"
                                        except:
                                            page = ""
                                        documents_content += (
                                            "**Source: "
                                            + str(document.metadata["source"])
                                            + page
                                            + "**\n\n"
                                        )
                                        documents_content += document.page_content + "\n\n\n"
                                    
                                    st.markdown(documents_content)
        else:
            # Regular RAG response
            if hasattr(st.session_state, 'chain'):
                # 1. Invoke LLM
                response = st.session_state.chain.invoke({"question": prompt})
                answer = response["answer"]

                # 2. Display results
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    # 2.1. Display answer:
                    st.markdown(answer)

                    # 2.2. Display source documents:
                    if "source_documents" in response and response["source_documents"]:
                        with st.expander("**Source documents**"):
                            documents_content = ""
                            for document in response["source_documents"]:
                                try:
                                    page = " (Page: " + str(document.metadata["page"]) + ")"
                                except:
                                    page = ""
                                documents_content += (
                                    "**Source: "
                                    + str(document.metadata["source"])
                                    + page
                                    + "**\n\n"
                                )
                                documents_content += document.page_content + "\n\n\n"

                            st.markdown(documents_content)
            else:
                # If no RAG chain is available, use direct LLM
                try:
                    llm = TogetherLLM(
                        api_key=API_KEY,
                        model_name=st.session_state.selected_model,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens,
                    )
                    answer = llm._call(prompt)
                    
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.chat_message("user").write(prompt)
                    st.chat_message("assistant").write(answer)
                    
                except Exception as e:
                    st.error(f"Error getting response from LLM: {str(e)}")

    except Exception as e:
        st.warning(f"Error: {str(e)}")

####################################################################
#                         Chatbot
####################################################################
def chatbot():
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("💬 Chat with your data & research papers")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message.get(
                    st.session_state.get("assistant_language", "english"), 
                    dict_welcome_message["english"]
                ),
            }
        ]
    
    # Initialize other session state variables
    if "enable_arxiv" not in st.session_state:
        st.session_state.enable_arxiv = True
    if "max_papers" not in st.session_state:
        st.session_state.max_papers = 5
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.5
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1000
    if "assistant_language" not in st.session_state:
        st.session_state.assistant_language = "english"
    if "retriever_type" not in st.session_state:
        st.session_state.retriever_type = "Contextual compression"

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input
    if prompt := st.chat_input():
        # Show helpful tips
        if not hasattr(st.session_state, 'chain'):
            st.info("💡 **Tips:**\n"
                   "- Upload documents and create a vectorstore to chat with your data\n"
                   "- Ask me to 'search for research papers on [topic]' to find arXiv papers\n"
                   "- I can answer general questions using Together AI models")
        
        with st.spinner("Processing..."):
            get_response_from_LLM(prompt=prompt)

    # Display some example prompts
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Example Prompts")
    st.sidebar.markdown("""
    **For Research Papers:**
    - "Search for recent research papers on machine learning"
    - "Find papers about climate change mitigation"
    - "Look up studies on artificial intelligence ethics"
    
    **For Document Chat:**
    - "What are the key points in my documents?"
    - "Summarize the main findings"
    - "What does the document say about [topic]?"
    
    **General Questions:**
    - "Explain quantum computing"
    - "What are the latest trends in AI?"
    - "How does blockchain work?"
    """)

# Create required directories if they don't exist
def ensure_directories():
    """Ensure required directories exist"""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    chatbot()