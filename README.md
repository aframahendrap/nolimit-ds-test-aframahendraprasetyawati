# Monopoly Chatbot
A specialized chatbot for the **Monopoly board game**, which answers user questions **strictly based on the official Monopoly rules PDF**.

---

## Dataset
- **Source**: [Official Monopoly Rule Book PDF](https://902231.app.netsuite.com/core/media/media.nl?id=488686&c=902231&h=9fc37d8b226fe536bbfe&_xt=.pdf)  
- **Annotation / Preprocessing**:  
  - Extracted text using `pdfplumber`.  
  - Each page segmented and stored with metadata (`page number` and `title`) for reference.  
  - Text embeddings generated using `sentence-transformers/all-MiniLM-L6-v2`.  
  - FAISS index built for fast similarity search.  
> All answers are constrained to the content of the PDF; the chatbot **does not use any external knowledge**.

---

## Setup Instructions
1. **Clone repository**  
   ```bash
   git clone <repo_url>
   cd <project_folder>

2. **Create and activate virtual environment**
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate

3. **Install dependencies**
    pip install -r requirements.txt

4. **Set up environment variables**
    Create a .env file in the project root: GOOGLE_API_KEY="your_google_api_key_here"

5. **Run the chatbot**
    # Command line interface: 
      python chatbot.py
    # Streamlit web interface:
      streamlit run ui_web.py

---

## Chosen Models
1. Embedding Model: sentence-transformers/all-MiniLM-L6-v2
2. LLM Model: gemini-2.5-flash (via Google Generative AI API)

---


