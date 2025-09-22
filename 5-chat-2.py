import streamlit as st
import lancedb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("data/lancedb")
    return db.open_table("cobol-token")


import json

def get_context(query: str, table, num_results: int = 5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    results = table.search(query).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Safely extract metadata
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):  # convert JSON string → dict
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        # Pull values with defaults
        filename = metadata.get("filename", "")
        page_numbers = metadata.get("page_numbers", [])
        title = metadata.get("title", "")

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}" if source_parts else ""
        if title:
            source += f"\nTitle: {title}"

        # Add chunk text + source
        text = row.get("text", "")
        contexts.append(f"{text}{source}")

    return "\n\n".join(contexts)

def get_chat_response(messages, context: str) -> str:
    """Get streaming response from OpenAI API.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer questions. If you're unsure or the context
    doesn't contain the relevant information, say so.
    
    Context:
    {context}
    """

    messages_with_context = [{"role": "system", "content": system_prompt}, *messages]
    print(f"messages_with_context: {messages_with_context}")

    # Create the streaming response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_context,
        temperature=0.7,
        stream=True,
    )

    # Use Streamlit's built-in streaming capability
    response = st.write_stream(stream)
    return response


# Initialize Streamlit app
st.title("📚 Document Q&A")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
table = init_db()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    # Get relevant context
with st.status("Searching document...", expanded=False) as status:
    results = table.search(prompt).limit(5).to_pandas()

    st.markdown(
        """
        <style>
        .search-result {
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
            background-color: #f0f2f6;
        }
        .search-result summary {
            cursor: pointer;
            color: #0f52ba;
            font-weight: 500;
        }
        .search-result summary:hover {
            color: #1e90ff;
        }
        .metadata {
            font-size: 0.9em;
            color: #666;
            font-style: italic;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.write("Found relevant sections:")

    contexts = []

    for _, row in results.iterrows():
        import json

        # Parse metadata safely
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        filename = metadata.get("filename", "Unknown file")
        page_numbers = metadata.get("page_numbers", [])
        title = metadata.get("title", "Untitled section")

        # Build standardized source string
        source_parts = [filename]
        if page_numbers:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")
        source = " - ".join(source_parts)

        # Show in UI
        st.markdown(
            f"""
            <div class="search-result">
                <details>
                    <summary>{source}</summary>
                    <div class="metadata">Section: {title}</div>
                    <div style="margin-top: 8px;">{row.get("text","")}</div>
                </details>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Build standardized context
        chunk_context = f"""{row.get("text","")}
Source: {source}
Title: {title}"""
        contexts.append(chunk_context)

    # Final context string for model
    context = "\n\n".join(contexts)

    # Display assistant response first
    with st.chat_message("assistant"):
        # Get model response with streaming
        response = get_chat_response(st.session_state.messages, context)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
