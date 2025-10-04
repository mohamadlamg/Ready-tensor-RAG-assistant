from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# #Preparation des données

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber

# texts = [
#     "papa aime manger le riz",
#     "maman aime le garba",
#     "momo aime le football",
#     "oumar aime le basketball"
# ]

# metadatas = [
#     {"topic":"nourriture","type":"preference"},
#     {"topic":"nourriture","type":"preference"},
#     {"topic":"sport","type":"football"},
#     {"topic":"sport","type":"basketball"}
# ]

# documents = [
#     Document(page_content=text, metadata=metadatas[i])
#     for i, text in enumerate(texts)
# ]

# #Stockage des vecteurs

from langchain_community.vectorstores import Chroma

# vectorstore = Chroma.from_documents(documents, embeddings)

# results = vectorstore.similarity_search_with_score("Quel sport fait papa ?", k=2)

# for doc, score in results:
#     print(f"Score: {score:.3f}")
#     print(f"Text: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")
#     print("---")

file = 'Réussir son site web en 60 fiches.pdf'
def process_document_file(file_path):
    # Read the document
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    

    # Split intelligently
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    # Create documents with metadata
    documents = [
        Document(
            page_content=chunk, 
            metadata={"source": file_path, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    # Create searchable vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings)

    results = vectorstore.similarity_search_with_score("Fais le bilan du document en 10 lignes", k=2)

    for doc, score in results:
        print(f"Score: {score:.3f}")
        print(f"Text: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("---")

    return vectorstore

process_document_file(file)