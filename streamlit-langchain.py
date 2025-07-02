import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import bs4  # for WebBaseLoader

# Load environment variables
load_dotenv()
token = os.getenv("SECRET")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# Load all document sources
file_loader = TextLoader("data/hometown_info.txt", encoding="utf-8")
file_docs = file_loader.load()

web_loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2017-06-21-overview/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
)
web_docs = web_loader.load()

wiki_loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Panevėžys",)
)
wiki_docs = wiki_loader.load()

# Function to filter sources based on user selection
def filter_docs(selected_sources):
    docs = []
    if "File" in selected_sources:
        docs += file_docs
    if "Web" in selected_sources:
        docs += web_docs
    if "Wikipedia" in selected_sources:
        docs += wiki_docs
    return docs

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

# Prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

# Formatter to separate source contents
def format_docs(docs):
    sources = {
        "File (hometown_info.txt)": [],
        "Wikipedia (wikipedia.org)": [],
        "Web (lilianweng.github.io)": []
    }
    for doc in docs:
        src = doc.metadata.get("source", "")
        if "hometown_info.txt" in src:
            sources["File (hometown_info.txt)"].append(doc.page_content)
        elif "lilianweng.github.io" in src:
            sources["Web (lilianweng.github.io)"].append(doc.page_content)
        elif "wikipedia.org" in src:
            sources["Wikipedia (wikipedia.org)"].append(doc.page_content)
    output = []
    for name, contents in sources.items():
        output.append(f"## Source: {name}\n" + ("\n\n".join(contents) if contents else "_No content found from this source._"))
    return "\n\n".join(output)

# Streamlit UI
st.title("My Hometown - LangChain Q&A")

with st.form("my_form"):
    text = st.text_area("Klausk apie miestą Panevėžys:", "Kokia miesto istorija?")

    sources_selected = st.multiselect(
        "Pasirink šaltinius:",
        ["File", "Web", "Wikipedia"],
        default=["File", "Web", "Wikipedia"]
    )

    submitted = st.form_submit_button("Submit")

if submitted:
    # Step 1: Filter documents
    selected_docs = filter_docs(sources_selected)
    splits = text_splitter.split_documents(selected_docs)

    # Step 2: Create vectorstore and retriever
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        ),
    )
    retriever = vectorstore.as_retriever()

    # Step 3: Set up LLM and RAG chain
    llm = ChatOpenAI(
        base_url=endpoint,
        temperature=0.7,
        api_key=token,
        model=model
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Step 4: Show result
    st.info(rag_chain.invoke(text))
