import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv 
import speech_recognition as sr
import asyncio

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# PDF validation function
def validate_pdf(file):
    if file.type == "application/pdf":
        if file.size <= 200 * 1024 * 1024:
            return True
        else:
            st.error("PDF file size exceeds 200 MB limit.")
            return False
    else:
        st.error("Please upload only PDF files.")
        return False
    

# URL validation function
def validate_url(url):
    if not url:
        return False
    elif url.startswith("http://") or url.startswith("https://"):
        return True
    else:
        st.error("Please enter a valid URL starting with 'http://' or 'https://'. ")
        return False


# Extract text from URL
def get_data_from_url(url):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        return ''.join([document.page_content for document in documents])
    except Exception as e:
        st.error(f"Error loading URL: {e}")
        return ""


# Extract text from pdf
def get_pdf_text(pdf):
    try:
        pdf_reader = PdfReader(pdf)
        return ''.join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


# Extract text from PDF
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)
    

# Create vector store
def get_vector_store(text_chunks, embeddings):
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")


# Get conversational chain
def get_conversational_chain():

    prompt_template = """
        You are a knowledgeable and polite assistant. Your task is to provide clear, detailed, and accurate answers based on the provided context. 
        If the information is not available in the context, respond with, "I'm sorry, but the answer is not available in the context provided."

        Context:
        {context}

        Question:
        {question}

        Answer the question by following these guidelines:
        1. If the answer is found directly in the context, provide a detailed and complete response, ensuring clarity and conciseness.
        2. If the context provides partial information, use it to construct a comprehensive answer while indicating the limitations.
        3. If the answer is not available in the context, clearly state, "I'm sorry, but the answer is not available in the context provided."
        4. Always maintain a polite and helpful tone in your responses.
        5. Begin your answer by acknowledging the question and summarizing it briefly.
        6. Where applicable, break down your response into clear sections or bullet points for better readability.
        7. Ensure that technical terms are explained in simple language when possible to aid user understanding.
        8. End your response by asking if further clarification is needed or if there are any follow-up questions.

        Answer:
    """


    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)



# Process inputs (PDFs and URLs)
def process_inputs(pdf_files, urls):
    raw_text = ""

    # process pdfs
    if pdf_files:
        for pdf in pdf_files:
            if validate_pdf(pdf):
                raw_text += get_pdf_text(pdf)
            else:
                return

    # process urls
    if urls:
        for url in urls:
            if validate_url(url):
                raw_text += get_data_from_url(url)
            else:
                return
        
    if not raw_text:
        st.error("No valid data provided. Please upload PDF files or enter valid URL.")
        return

    text_chunks = get_text_chunks(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    get_vector_store(text_chunks, embeddings)
    


# Handle user input query
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


# Handle voice input
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            st.write("Processing voice input...")
            query = recognizer.recognize_google(audio)
            return query
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand what you said.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
        except sr.WaitTimeoutError:
            st.error("Listening timed out while waiting for phrase to sttart")
    return ""
    


# Main interface
def main():
    st.set_page_config("Chat With Multiple PDF and URL", layout="wide")
    st.header("Chat with Multiple PDF and Websites using ChatFlex ")

    if "query_text" not in st.session_state:
        st.session_state.query_text = ""

    
    user_question = st.text_input("Ask a Question from the PDF Files or URL", value=st.session_state.query_text)

    # voice input
    if st.button("Record", key="record_button", help="Click to record voice"):
        query = voice_input()
        if query:
            st.session_state.query_text = query
            st.rerun()

        
    if user_question:
            user_input(user_question)

    with st.sidebar:
        st.title("Menu:")

        # combine pdf uploader and url input
        pdf_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        num_urls = st.number_input("Number of URLs", min_value=1, max_value=10, value=1)

        urls = []
        for i in range(num_urls):
            url = st.text_input(f"URL {i+1}")
            urls.append(url)

        if st.button("Submit & Process") and (pdf_files or urls):
            with st.spinner("Processing..."):
                process_inputs(pdf_files, urls)
                


if __name__ == "__main__":
    main()