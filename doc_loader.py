import os
import tempfile
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,                    # For PDF
    UnstructuredWordDocumentLoader,  # For DOCX
    UnstructuredEmailLoader          # For .eml
)


from dotenv import load_dotenv
load_dotenv()

def load_and_split_documents(file_objects):
    """
    Load documents from given file objects (PDF, DOCX, EML), split them into chunks.
    Adds source file name and page number (for PDFs) to metadata.
    """
    all_docs = []

    for file in file_objects:
        suffix = file.filename.split(".")[-1].lower()
        file_name = file.filename

        # Save temporarily
        temp = tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix)
        file.save(temp.name)

        try:
            if suffix == "pdf":
                loader = PyMuPDFLoader(temp.name)
            elif suffix == "docx":
                loader = UnstructuredWordDocumentLoader(temp.name)
            elif suffix == "eml":
                loader = UnstructuredEmailLoader(temp.name)
            else:
                print(f"Unsupported file type: {file_name}")
                continue

            docs = loader.load()

            # Add file name and page number (if any) to metadata
            for doc in docs:
                doc.metadata["source_file"] = file_name
                if "page_number" not in doc.metadata:
                    doc.metadata["page_number"] = None

            all_docs.extend(docs)

        except Exception as e:
            print(f"Failed to load {file_name}: {e}")
        finally:
            temp.close()
            os.unlink(temp.name)

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    return chunks


# def create_and_save_faiss_vectorstore(chunks, output_dir="store"):
#     """
#     Embed and save the FAISS vector store with metadata.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     embedder =SentenceTransformer('all-MiniLM-L6-v2')

#     vectorstore = FAISS.from_documents(chunks, embedder)
#     vectorstore.save_local(output_dir)

#     with open(os.path.join(output_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
