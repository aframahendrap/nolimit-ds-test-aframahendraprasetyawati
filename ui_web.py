import streamlit as st
from chatbot import extract_text, build_faiss, search_and_answer, PDF_PATH

# Load data
st.set_page_config(page_title="Monopoly Chatbot", page_icon="👾")

# Custom style
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
        color: #333333;
    }
    /* User chat bubble */
    .user-message {
        background-color: #d1e7dd;
        border-radius: 12px;
        padding: 8px;
        margin: 4px 0;
        max-width: 85%;
    }
    /* Assistant chat bubble */
    .assistant-message {
        background-color: #f8d7da;
        border-radius: 12px;
        padding: 8px;
        margin: 4px 0;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-width: 85%;
    }
    .stExpanderHeader {
        color: #0d6efd;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_index():
    texts = extract_text(PDF_PATH)
    index, texts = build_faiss(texts)
    return index, texts


index, texts = load_index()

# UI
st.title("Monopoly Chatbot")
st.write(
    "Ask questions about the Monopoly game rules and get answers based on the official rulebook."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-message">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-message">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )


# User input
if query := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user message
    st.markdown(f'<div class="user-message">🧑{query}</div>', unsafe_allow_html=True)

    # Assistant answer
    with st.spinner("Thinking..."):
        answer, sources = search_and_answer(query, index, texts)

        st.markdown(
            f'<div class="assistant-message">💭{answer}</div>', unsafe_allow_html=True
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # sumber
        with st.expander("Sources"):
            for s in sources:
                st.markdown(
                    f"<span style='color:#0d6efd;'>{s}</span>", unsafe_allow_html=True
                )
