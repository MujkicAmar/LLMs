import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


model = "tinyllama"

files = glob.glob("DOCUMENTS/*.pdf")

pdf_documents = []
for document in files: 
    loader = PyPDFLoader(document)
    docs = loader.load()
    pdf_documents.extend(docs)


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pdf_documents)

for chunk in chunks:
    if "Norfolk Southern" in chunk.page_content:
        print(chunk)