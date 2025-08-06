import asyncio
# Create and set a new event loop if none exists in the current thread
try:
    asyncio.get_running_loop()
except RuntimeError:  # No event loop in this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
import re
import streamlit as  st
from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import os
import json

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import traceback

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from dotenv import load_dotenv

load_dotenv()

import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def clean_chunk_text(text):
    # Removes lines like: "3) Match the followings:" or "4) Enlist the features..."
    return re.sub(r"\n?\d+\)[^\n]*", "", text)

def detect_section(text):
    """
    Very basic logic to extract a potential section title from the beginning of a text chunk.
    Assumes headings are lines in ALL CAPS or numbered like 1., 1.1, etc.
    """
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if re.match(r'^([A-Z\s]{5,}|(\d+\.\s|[A-Z][a-z]+\s)+)$', line):
            return line
    return "Unknown Section"

def prepare_documents(chunks, source_name):
    documents = []
    for chunk in chunks:
        cleaned_chunk = clean_chunk_text(chunk)  # Clean the chunk here
        section_title = detect_section(cleaned_chunk)  # <- write this function!
        doc = Document(page_content=cleaned_chunk, metadata={"source": source_name, "section": section_title})
        documents.append(doc)
    return documents

def get_pdf_text(pdf_docs):
    text = ""
    # If single file, convert to list
    if not isinstance(pdf_docs, (list, tuple)):
        pdf_docs = [pdf_docs]

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(documents):
    # Convert Document objects to dicts with only serializable fields
    serializable_docs = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    # Use the actual Document objects to build vector store
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


from langchain.chains.llm import LLMChain

def get_conversational_chain():
    from langchain.prompts import PromptTemplate
    from langchain.chains.question_answering import load_qa_chain
    from langchain_google_genai import ChatGoogleGenerativeAI

    initial_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use the following context to answer the question concisely and clearly.

Limit your answer to no more than 2-3 sentences, summarizing key relevant points from the context.
If the answer is unknown from the context, respond with "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=initial_prompt,
    )
    return chain

def user_input(user_question): 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))   
    vector_store = FAISS.load_local(
        folder_path="C:/Users/hp/Documents/Chat with multi PDF's/faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    ) 
    docs = vector_store.similarity_search(user_question, k=1)

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following context to answer the question concisely and clearly.

Limit your answer to no more than 2-3 sentences, summarizing key relevant points from the context.
If the answer is unknown from the context, respond with "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)


    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=initial_prompt,
    )  
    response = chain.run(input_documents=docs, question=user_question)

    st.write("Reply:", response)

def load_vector_store_if_exists():
            if os.path.exists("faiss_index"):
                embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
                return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            return None

def main():
    st.set_page_config(page_title="Chat With Multi PDF")
    st.header("Chat with Multi PDF using Gemini💁")

    # Initialize session state variables if not already present
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store_if_exists()
        st.session_state.processed = st.session_state.vector_store is not None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "pdf_list" not in st.session_state:
        st.session_state.pdf_list = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and click on Submit & Process", accept_multiple_files=True, key="pdf_uploader"
        )
        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                all_documents = []
                pdf_names = []

                for pdf_doc in pdf_docs:
                    raw_text = get_pdf_text(pdf_doc)  # process each PDF separately
                    text_chunks = get_text_chunks(raw_text)

                    pdf_file_name = pdf_doc.name

                    documents = prepare_documents(text_chunks, pdf_file_name)
                    all_documents.extend(documents)
                    pdf_names.append(pdf_file_name)

                # Build vector store with all documents (from all PDFs)
                vector_store = get_vector_store(all_documents)
                st.session_state.vector_store = vector_store
                st.session_state.processed = True
                st.session_state.pdf_list = pdf_names

            st.success("Done processing PDFs!")
            # Add these lines here to debug the session state after processing
            st.write(f"Processed: {st.session_state.processed}")
            st.write(f"PDFs loaded: {st.session_state.pdf_list}")
        #  Add this to clear chat along with vector store 
        if st.button("Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.vector_store = None
            st.session_state.processed = False
            st.session_state.pdf_list = []
            st.success("Chat and uploaded PDFs have been cleared.")


    # Only allow question input if PDFs are processed
    if st.session_state.processed:
    # Add dropdown to select PDF source here
        selected_pdf = st.selectbox("Select document", st.session_state.pdf_list)
        user_question = st.text_input("Ask a Question from the processed PDF files:")

        if user_question:
            if st.session_state.vector_store is None:
                st.error("Please upload and process PDFs first.")
            else:
               # Get relevant docs
                # Retrieve more docs initially for filtering
                all_docs = st.session_state.vector_store.max_marginal_relevance_search(
    query=user_question, k=10, fetch_k=20
)

                # Filter docs to only those with source (filename) matching selected_pdf
                filtered_docs = [doc for doc in all_docs if doc.metadata.get("source") == selected_pdf]

                st.write(f"Docs retrieved (before filtering): {len(all_docs)}")
                st.write(f"Docs after filtering by selected PDF ({selected_pdf}): {len(filtered_docs)}")
                for i, d in enumerate(filtered_docs):
                    st.write(f"Doc {i}: {d.metadata.get('source', 'Unknown')} - snippet: {d.page_content[:100]}")

                if not filtered_docs:
                    cleaned_answer = "Sorry, no relevant content found in the selected document."
                else:
                    try:
                        # Build conversational context
                        history_context = ""
                        for chat in st.session_state.chat_history[-3:]:  # last 3 turns only
                            history_context += f"User: {chat['user']}\nAssistant: {chat['bot']}\n"
                        full_question = f"{history_context}User: {user_question}"

                        chain = get_conversational_chain()
                        response = chain.run(input_documents=filtered_docs, question=full_question)

                        def clean_answer_text(text):
                            return re.sub(r'\d+\)\s.*', '', text).strip()

                        cleaned_answer = clean_answer_text(response.strip())
                    except Exception:
                        error_msg = traceback.format_exc()
                        st.error("An error occurred while generating answer. See console for details.")
                        print(error_msg)
                        cleaned_answer = "Sorry, an error occurred while generating the answer."

            
                # Append to chat history only if response is not same as last one
                if not st.session_state.chat_history or cleaned_answer != st.session_state.chat_history[-1]["bot"]:
                    st.session_state.chat_history.append({"user": user_question, "bot": cleaned_answer})


                # Display chat history
                for chat in st.session_state.chat_history:
                    st.markdown(f"**You:** {chat['user']}")
                    st.markdown(f"**Bot:** {chat['bot']}")


        else:
            st.info("Upload and process PDF files first to start asking questions.")




if __name__ == "__main__":
    main()


