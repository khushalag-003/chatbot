import streamlit as st
from PyPDF2 import PdfReader
# split the data into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
# Facebook AI Research, Faiss is an open-source library for fast, dense vector similarity search and grouping
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
OPENAI_API_KEY = "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"

# upload pdf file
st.header("Chatbot")

with st.sidebar:
    st.title("Your Document")
    file = st.file_uploader("Upload a PDF file and start asking questions ", type="pdf")

# Extract the text

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

# Break into chunks

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    #generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating vector stores
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user input
    user_question = st.text_input("Type your question here")

    #do similarity searchSS
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            #temperature defines randomness
            temperature=0,
            max_tokens =1000,
            model_name="gpt-3.5-turbo"




        )
        #stuff basically mean stuff al the data into a bucket and pass it to llm
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)



