import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai


load_dotenv()

# Setup 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_PATH = "data/monopoly.pdf"
genai.configure(api_key=GOOGLE_API_KEY)
FAISS_INDEX_PATH = "vectorstores/monopoly.index"

# Load models 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)
# LLM using gemini-2.5-flash 
LLM_MODEL = "gemini-2.5-flash"


# Extract text from PDF
def extract_text(pdf_path):
    """Extract text from PDF and return as list of (page_number, text) tuples."""
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                lines = text.strip().split("\n")

                # Title for source reference
                title = " "
                for line in lines:
                    if line.isupper() or (len(line.split()) <= 5 and line.istitle()):
                        title = line.strip()
                        break
                texts.append(
                    {"page": f"Page {i+1}", "title": title, "content": text.strip()}
                )
    print(f"[INFO] Extracted {len(texts)} pages from PDF")
    return texts


# Build FAISS index
def build_faiss(texts):
    """Build FAISS index from list of dicts."""
    contents = []
    for item in texts:
        contents.append(item["content"])

    embeddings = embedder.encode(contents)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    print(f"[INFO] FAISS index created with {len(texts)} entries")
    return index, texts


# Search and answer with LLM
def search_and_answer(query, index, texts, top_k=3):
    """Search top-k FAISS index and answer using Gemini LLM directly."""
    query_embedding = embedder.encode([query])
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), top_k
    )

    retrieved_docs = []
    retrieved_sources = []

    for idx in indices[0]:
        retrieved_docs.append(texts[idx]["content"])
        page = texts[idx]["page"]
        title = texts[idx]["title"]
        retrieved_sources.append(f"{page} - {title}")

    retrieved_sources = list(set(retrieved_sources))
    context = "\n\n".join(retrieved_docs)

    prompt_text = f"""
    You are a specialized Q&A assistant for the Monopoly board game. 
    Your knowledge is strictly limited to the rule document provided in the context.

    Your task is to provide an answer that captures the **meaning and facts** from the document relevant to the user's question. 
    You may rephrase or adjust the **style and wording** of the answer for readability or natural flow, 
    but the **facts, rules, and details** must remain exactly as in the document. 
    Do not add, omit, or assume anything that is not present in the document.

    Follow these rules precisely:

    1. **Find by Meaning with Synonyms**: Carefully locate the sentences or paragraph from the context that directly answer the question. 
        You must also recognize **synonyms, paraphrases, or alternative expressions** in the user's question that refer to the same concept in the document.

    2. **Faithful Rephrasing Only**: You may rewrite sentences for clarity, conciseness, or natural flow, 
        but the **content and all details must remain faithful** to the original document.

    3. **Handle "Not Found"**: If no relevant answer exists in the context, reply in the same language as the user's "Question" 
        with a message stating that the answer was not found in the document.

    Context:
    ---
    {context}
    ---

    Question: {query}
"""

    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt_text)

    return response.text, retrieved_sources


# Main 
if __name__ == "__main__":
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = extract_text(PDF_PATH)
    index, texts = build_faiss(texts)

    print("\nMonopoly Chatbot Ready! Ask anything (type 'exit' to quit)\n")
    while True:
        query = input("Your Question: ")
        if query.lower() in ["exit"]:
            break
        answer, sources = search_and_answer(query, index, texts)
        print("\nAnswer:", answer)
        print("Sources:", sources, "\n")
