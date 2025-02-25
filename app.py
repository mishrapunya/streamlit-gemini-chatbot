import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
import docx
import re

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def read_pdf(file):
    """Extract text from PDF files"""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    """Extract text from DOCX files"""
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def read_txt(file):
    """Read text from TXT files"""
    return file.getvalue().decode("utf-8")

def extract_text_from_file(file):
    """Extract text based on file type"""
    if file.name.endswith('.pdf'):
        return read_pdf(file)
    elif file.name.endswith('.docx'):
        return read_docx(file)
    elif file.name.endswith('.txt'):
        return read_txt(file)
    else:
        return "Unsupported file format"

def get_gemini_response(prompt, context="", model="gemini-1.5-pro", temperature=0.7):
    """Get response from Gemini model with temperature control"""
    model = genai.GenerativeModel(model, generation_config={"temperature": temperature})
    response = model.generate_content(f"{context}\n\nUser Query: {prompt}")
    return response.text

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "context" not in st.session_state:
        st.session_state.context = ""
    
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    
    if "bot_name" not in st.session_state:
        st.session_state.bot_name = "Gemini Assistant"
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

def main():
    # Initialize session state
    initialize_session_state()
    
    st.set_page_config(page_title="Customizable Gemini Chatbot", page_icon="🤖")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Bot Configuration")
        
        # API Key input
        api_key = st.text_input("Google Gemini API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
        
        # Bot name
        bot_name = st.text_input("Bot Name", value=st.session_state.bot_name)
        if bot_name != st.session_state.bot_name:
            st.session_state.bot_name = bot_name
        
        # System prompt
        st.subheader("System Prompt")
        system_prompt = st.text_area(
            "Enter a system prompt to define your bot's behavior",
            value=st.session_state.system_prompt,
            height=150
        )
        if system_prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = system_prompt
        
        # Temperature slider (NEW)
        st.subheader("AI Settings")
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make output more creative, lower values make it more focused and deterministic"
        )
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature
        
        # Upload reference documents
        st.subheader("Reference Documents")
        uploaded_files = st.file_uploader(
            "Upload documents for context (PDF, DOCX, TXT)",
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Documents"):
            context_text = ""
            for file in uploaded_files:
                file_content = extract_text_from_file(file)
                context_text += f"\n\n--- Content from {file.name} ---\n{file_content}"
            
            st.session_state.context = context_text
            st.success(f"Processed {len(uploaded_files)} document(s)")
        
        # Model selection
        model_option = st.selectbox(
            "Select Gemini Model",
            ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        )
        
        # Reset chat button
        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.info("📝 Configure your bot on the left sidebar, then chat below!")
    
    # Main chat interface
    st.title(f"Chat with {st.session_state.bot_name}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        # Check if API key is provided
        if not api_key:
            st.error("Please enter your Google Gemini API Key in the sidebar")
            st.stop()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare full context for Gemini
        full_context = f"System Instructions: {st.session_state.system_prompt}\n\n"
        if st.session_state.context:
            full_context += f"Reference Information: {st.session_state.context}\n\n"
        full_context += "Previous Messages:\n"
        for msg in st.session_state.messages[:-1]:  # Exclude the latest message
            full_context += f"{msg['role'].title()}: {msg['content']}\n"
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_gemini_response(
                    prompt, 
                    full_context, 
                    model_option, 
                    st.session_state.temperature
                )
                st.markdown(response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
