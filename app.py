import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
import requests
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From Website")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from Website"):
    # Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load website data using requests and BeautifulSoup
                response = requests.get(generic_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    content = soup.get_text(strip=True)
                    logging.debug(f"Retrieved content: {content[:500]}...")  # Log first 500 characters
                else:
                    content = ""
                    logging.error(f"Failed to retrieve content from URL: {generic_url}")
                
                # Check if content is empty
                if not content.strip():
                    st.error(f"No content was retrieved from the provided URL ({generic_url}). Please check the URL and try again.")
                else:
                    # Prepare the documents for summarization
                    documents = [Document(page_content=content)]

                    # Chain For Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run({"input_documents": documents})

                    st.success(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error("Error occurred", exc_info=True)