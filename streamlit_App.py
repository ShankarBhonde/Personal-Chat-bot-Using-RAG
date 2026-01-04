import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
<style>
#normal-blink {
    text-align: center;
    background: rgba(0, 0, 0, 0.75);
    padding: 5px 30px;
    border-radius: 12px;
    border: 3px solid #f28705;
    backdrop-filter: blur(5px);
    margin-top: 20px;
}
.blink {
    font-size: 35px;
    font-weight: bold;
    color: #00C4FF;
    animation: blinker 1s linear infinite;
}
@keyframes blinker {
    50% { opacity: 0; }
}
</style>

<div id="normal-blink">
    <div class="blink">Welcome To My Chatbot 💁</div>
</div>
""", unsafe_allow_html=True)

# ------------------ API KEY ------------------
api_key = st.text_input("Enter your Google API Key:", type="password")

# ------------------ PDF FUNCTIONS ------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# ------------------ QA CHAIN (NEW VERSION) ------------------
def get_conversational_chain(api_key):
    prompt = PromptTemplate(
        template="""
Answer the question as detailed as possible using ONLY the provided context.
If the answer is not present, say:
"Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    return create_stuff_documents_chain(model, prompt)


# ------------------ USER QUERY ------------------
def user_input(user_question, api_key):
    if not os.path.exists("faiss_index"):
        st.error("⚠ Please upload and process PDF first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(user_question, k=4)
    chain = get_conversational_chain(api_key)

    response = chain.invoke({
        "context": docs,
        "question": user_question
    })

    st.subheader("Reply")
    st.write(response)


# ------------------ MAIN APP ------------------
def main():
    st.header("AI Clone Chatbot 💁")

    user_question = st.text_input("Ask a question from the PDF files")
    ask_btn = st.button("Submit Question")

    if ask_btn:
        if not api_key:
            st.error("⚠ Please enter API key.")
        elif not user_question:
            st.error("⚠ Please enter a question.")
        else:
            user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process") and api_key:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks, api_key)
                st.success("✅ PDFs processed successfully")


if __name__ == "__main__":
    main()
