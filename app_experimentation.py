import sys
import os

try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
import numpy as np
from datetime import datetime
import pandas as pd
import json
import google.generativeai as genai
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions

# Handle missing API key safely
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please configure it in your Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Add MongoDB configuration near the top of the file
# Use st.secrets to store your MongoDB connection string
MONGO_CONNECTION_STRING = st.secrets.get("MONGO_CONNECTION_STRING")
if not MONGO_CONNECTION_STRING:
    st.error("MongoDB connection string not found. Please configure it in your Streamlit secrets.")
    st.stop()

# Initialize ChromaDB client
@st.cache_resource
def init_chroma():
    try:
        # Create a persistent directory for ChromaDB
        os.makedirs("chroma_db", exist_ok=True)
        
        # Initialize the client with persistence
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        
        # Use ChromaDB's default embedding function
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        return chroma_client, embedding_function
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        raise e

# Add this function to handle collection creation and data loading
@st.cache_resource
def init_qa_collection(_chroma_client, _embedding_function, collection_name="msds_program_qa_labeled"):
    try:
        # Delete existing collection if it exists
        try:
            _chroma_client.delete_collection(name=collection_name)
            if st.session_state.get('debug_mode', False):
                st.info(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            # Collection might not exist, which is fine
            pass
            
        # Create new collection
        qa_collection = _chroma_client.create_collection(
            name=collection_name,
            embedding_function=_embedding_function
        )

        # Load QA data with more detailed error handling
        try:
            # Change to labeled_qa.csv
            qa_df = pd.read_csv("labeled_qa.csv", on_bad_lines='warn')
            
            # Verify required columns exist
            required_columns = ['Category', 'Question', 'Answer']
            missing_columns = [col for col in required_columns if col not in qa_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Display DataFrame info for debugging only in debug mode
            if st.session_state.get('debug_mode', False):
                st.write("CSV columns:", qa_df.columns.tolist())
                st.write("First few rows:", qa_df.head())
            
            # Add data to the collection with correct column order
            qa_collection.add(
                ids=[str(i) for i in qa_df.index.tolist()],
                documents=qa_df['Question'].tolist(),
                metadatas=[{
                    'Answer': row['Answer'],
                    'Category': row['Category']
                } for _, row in qa_df.iterrows()]
            )

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.error("Please ensure labeled_qa.csv has these columns: Category, Question, Answer")
            raise e

        return qa_collection
    except Exception as e:
        st.error(f"Error initializing QA collection: {str(e)}")
        raise e

# Initialize ChromaDB and collection
try:
    chroma_client, embedding_function = init_chroma()
    qa_collection = init_qa_collection(chroma_client, embedding_function)
except Exception as e:
    st.error(f"Failed to initialize ChromaDB: {str(e)}")
    st.stop()

# Configure Gemini model
@st.cache_resource
def load_gemini_model():
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

gemini_model = load_gemini_model()

# Initialize MongoDB client
@st.cache_resource
def init_mongodb():
    try:
        # Add SSL and connection pool configurations
        client = MongoClient(
            MONGO_CONNECTION_STRING,
            tls=True,
            tlsAllowInvalidCertificates=False,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            retryWrites=True,
            maxPoolSize=50
        )
        
        # Test the connection
        client.admin.command('ping')
        
        db = client.MSDSchatbot
        return db.conversations
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

conversations_collection = init_mongodb()

# Modify save_conversation to handle None collection and store metrics
def save_conversation(session_id, user_message, bot_response, response_time, metrics=None):
    if conversations_collection is None:
        st.error("MongoDB connection not available")
        return
        
    try:
        conversation = {
            "session_id": session_id,
            "timestamp": datetime.now(),
            "user_message": user_message,
            "bot_response": bot_response,
            "feedback": None,
            "similarity_score": st.session_state.debug_similarity,
            "matched_question": st.session_state.debug_matched_question,
            "response_time_seconds": response_time
        }
        
        # Add metrics if available
        if metrics:
            conversation["metrics"] = metrics
            
        result = conversations_collection.insert_one(conversation)
        return str(result.inserted_id)
    except Exception as e:
        st.error(f"Error saving conversation to MongoDB: {str(e)}")
        return None

# Add this function after save_conversation
def update_feedback(conversation_id, feedback_type, details=None):
    if conversations_collection is None:
        st.error("MongoDB connection not available")
        return
        
    try:
        feedback_data = {
            "feedback_type": feedback_type,
            "timestamp": datetime.now()
        }
        
        # Add additional details if provided
        if details:
            feedback_data.update(details)
        
        conversations_collection.update_one(
            {"_id": conversation_id},
            {"$set": {
                "feedback": feedback_data,
                "last_updated": datetime.now()
            }}
        )
    except Exception as e:
        st.error(f"Error updating feedback: {str(e)}")

# Find the most similar question using ChromaDB
def find_most_similar_question(user_input, similarity_threshold=0.3):
    try:
        if qa_collection.count() == 0:
            return [], [], 0.0
        
        # Update to use the new return values
        processed_input, primary_category, all_categories = preprocess_query(user_input)
        
        # Query with filter for matching category
        results = qa_collection.query(
            query_texts=[processed_input],
            n_results=5,  # Get top 5 results
            where={"Category": {"$in": all_categories}} if "Other" not in all_categories else None
        )
        
        if not results['documents'][0]:
            return [], [], 0.0
        
        # Collect all questions and answers that meet the threshold
        matching_questions = []
        matching_answers = []
        best_similarity = 0.0
        
        for i, distance in enumerate(results['distances'][0]):
            similarity = 1 - distance
            if similarity >= similarity_threshold:
                matching_questions.append(results['documents'][0][i])
                matching_answers.append(results['metadatas'][0][i]['Answer'])
                best_similarity = max(best_similarity, similarity)
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            st.write("Top matches:")
            st.write(f"Categories: {all_categories}")
            st.write(f"Number of matching questions: {len(matching_questions)}")
            for i, (q, a, dist) in enumerate(zip(matching_questions, matching_answers, results['distances'][0])):
                sim = 1 - dist
                st.write(f"{i+1}. Question: {q}")
                st.write(f"   Similarity: {sim:.3f}")
                st.write(f"   Answer: {a[:100]}...")  # Show first 100 chars of answer
        
        return matching_questions, matching_answers, best_similarity
            
    except Exception as e:
        st.error(f"Error in find_most_similar_question: {str(e)}")
        return [], [], 0.0

# Enhance the preprocess_query function
def preprocess_query(query):
    processed_query = query.lower().strip()
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Analyze this question and return up to THREE most relevant categories from the following list, ordered by relevance:
        - Application Process: Questions about how to apply, deadlines, interviews, and application components
        - Admission Requirements: Questions about prerequisites, qualifications, and requirements
        - Financial Aid & Scholarships: Questions about funding, scholarships, and financial assistance
        - International Students: Questions specific to international student needs
        - Enrollment Process: Questions about post-acceptance procedures
        - Program Structure: Questions about program duration, format, and class sizes
        - Program Overview: Questions about general program information and features
        - Tuition & Costs: Questions about program costs, fees, and expenses
        - Program Preparation: Questions about preparing for the program
        - Faculty & Research: Questions about professors and research opportunities
        - Student Employment: Questions about work opportunities during the program
        - Student Services: Questions about health insurance and student support
        - Curriculum: Questions about courses and academic content
        - Practicum Experience: Questions about industry projects and partnerships
        - Career Outcomes: Questions about job placement, salaries, and career paths
        - Admission Statistics: Questions about typical GPAs, backgrounds, and work experience
        - Other: Questions that don't clearly fit into any of the above categories
        
        Examples:
        Question: "What GRE score do I need as an international student?" -> ["Application Process", "International Students"]
        Question: "How much is tuition and what scholarships are available?" -> ["Tuition & Costs", "Financial Aid & Scholarships"]
        Question: "Can I work while taking classes in the program?" -> ["Student Employment", "Program Structure"]
        Question: "Where is the nearest coffee shop?" -> ["Other"]
        
        Your question: "{query}"
        
        Return only the category names in a comma-separated list, nothing else."""
        
        response = model.generate_content(prompt)
        categories = [cat.strip() for cat in response.text.split(',')]
        primary_category = categories[0] if categories else "Other"
        
        if st.session_state.get('debug_mode', False):
            st.write(f"Detected categories: {categories}")
            
        return processed_query, primary_category, categories
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error categorizing query: {str(e)}")
        return processed_query, "Other", ["Other"]

def get_conversation_history(max_messages=5):
    """Get the recent conversation history formatted for the prompt"""
    if 'chat_history' not in st.session_state:
        return ""
    
    # Get last 5 message pairs (10 messages total)
    recent_messages = st.session_state.chat_history[-max_messages*2:]
    
    if not recent_messages:
        return ""
    
    # Format conversation history
    history = "\nRecent conversation history:\n"
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    return history

def get_gemini_response(user_input, retrieved_questions=None, retrieved_answers=None):
    try:
        # Load general information and context
        general_info = open('general_info.txt', 'r').read()
        context_data = json.load(open('context.json', 'r'))
        
        # Get conversation history
        conversation_history = get_conversation_history()
        
        # Process the query to get categories
        processed_query, primary_category, all_categories = preprocess_query(user_input)
        
        # Get category-specific information from context.json
        category_info = {}
        for category in all_categories:
            if category in context_data:
                category_info[category] = context_data[category]
        
        # Format QA pairs
        relevant_qa_pairs = ""
        if retrieved_questions and retrieved_answers and st.session_state.debug_similarity >= 0.3:
            if not isinstance(retrieved_questions, list):
                retrieved_questions = [retrieved_questions]
                retrieved_answers = [retrieved_answers]
            
            relevant_qa_pairs = "\n\nRelevant QA pairs from our database:\n"
            for q, a in zip(retrieved_questions, retrieved_answers):
                relevant_qa_pairs += f"Q: {q}\nA: {a}\n"
        
        # Enhanced prompt with conversation history
        prompt = f"""You are a helpful and friendly assistant for the University of San Francisco's MSDS program.
        
        Conversation History: {conversation_history}
        
        Current user question: "{user_input}"
        Primary Category: {primary_category}
        Related Categories: {', '.join(all_categories[1:]) if len(all_categories) > 1 else 'None'}
        
        Relevant information from all categories:
        ```
        {json.dumps(category_info, indent=2)}
        ```
        
        {relevant_qa_pairs}
        
        Instructions:
        1. Consider the conversation history when formulating your response
        2. If the user refers to previous messages, use that context
        3. Use ALL the provided QA pairs to formulate a comprehensive response
        4. If the QA pairs contain specific facts, numbers, or requirements, preserve them exactly
        5. Focus on answering the user's specific question
        6. Use a conversational tone while maintaining accuracy
        7. If any information is missing or unclear, acknowledge it
        
        Additional context:
        {general_info}
        
        Please provide your response:"""

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response."

# Get bot response (modified to include metrics)
def get_bot_response(user_input):
    if not user_input.strip():
        return "Please enter a question.", None
    
    # Update to use the new return values
    processed_query, primary_category, all_categories = preprocess_query(user_input)
    
    # Get all matching questions and answers
    matched_questions, matched_answers, similarity = find_most_similar_question(user_input)
    
    # Debug information
    st.session_state.debug_similarity = similarity
    
    # Show all matched questions
    if matched_questions:
        st.session_state.debug_matched_question = "\n".join([f"{i+1}. {q}" for i, q in enumerate(matched_questions)])
    else:
        st.session_state.debug_matched_question = "No match found"
    
    # Show all matched answers
    if matched_answers:
        st.session_state.debug_matched_answer = "\n".join([f"{i+1}. {a}" for i, a in enumerate(matched_answers)])
    else:
        st.session_state.debug_matched_answer = "No answer found"
    
    st.session_state.debug_category = f"{primary_category} (Related: {', '.join(all_categories[1:])})" if len(all_categories) > 1 else primary_category
    # Generate response using Gemini, passing all matched Q&As
    bot_response = get_gemini_response(user_input, matched_questions, matched_answers)
    
    return bot_response

def main():
    st.title("USF MSDS Program Chatbot")
    
    # Initialize session state variables
    for key in ['debug_matched_question', 'debug_matched_answer', 'debug_similarity', 
                'chat_history', 'session_id', 'conversation_ids', 'debug_category']:
        if key not in st.session_state:
            st.session_state[key] = "" if key not in ['chat_history', 'conversation_ids'] else []
            if key == 'debug_similarity':
                st.session_state[key] = 0.0
            elif key == 'session_id':
                st.session_state[key] = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Add to the session state initialization
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.now()

    tab1, tab2, tab3 = st.tabs(["Chat", "About", "Debug"])

    with tab1:
        # Add custom CSS for better layout
        st.markdown("""
            <style>
            /* Reduce space between title and input */
            .main > div:first-child {
                padding-bottom: 0rem;
            }
            
            /* Adjust input box and container spacing */
            .stTextInput {
                padding-bottom: 1rem;
            }
            
            /* Move chat container up */
            [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
                gap: 0rem;
                padding-top: 0rem;
            }
            
            /* Custom container styling */
            .chat-container {
                height: calc(100vh - 400px);
                min-height: 300px;
                overflow-y: auto;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 1rem;
                margin-top: 1rem;
                background-color: transparent;
                display: flex;
                flex-direction: column;
            }
            
            /* Create a messages wrapper for proper ordering */
            .messages-wrapper {
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
                min-height: 100%;
            }
            
            /* Ensure messages stack properly */
            .message {
                max-width: 80%;
                margin: 0.4rem 0;
                padding: 0.8rem 1rem;
                border-radius: 15px;
                word-wrap: break-word;
            }
            
            .user-message {
                background-color: #007AFF;
                color: white;
                margin-left: 20%;
                margin-right: 1rem;
            }
            
            .bot-message {
                background-color: #E9ECEF;
                color: black;
                margin-right: 20%;
                margin-left: 1rem;
            }
            
            /* Add loading animation */
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.3; }
                100% { opacity: 1; }
            }
            
            .loading {
                animation: pulse 1.5s infinite;
                padding: 10px;
                color: #666;
            }
            
            /* Improve message spacing */
            .message {
                margin: 0.8rem 0;
                padding: 0.8rem 1rem;
                border-radius: 15px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            
            /* Add hover effect on messages */
            .message:hover {
                box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            }
            
            /* Improve input field */
            .stTextInput input {
                border-radius: 20px;
                padding: 10px 15px;
            }
            
            /* Style the send button */
            .stButton button {
                border-radius: 20px;
                padding: 0.3rem 1.5rem;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create two columns for main chat and sidebar
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Input area
            st.subheader("Ask me about USF's MSDS program")
            user_message = st.text_input("Type your question here:", key="user_input")
            
            # Send button in the same line as input
            if st.button("Send", key="send_button") and user_message:
                with st.spinner("Thinking..."):
                    start_time = datetime.now()
                    bot_response = get_bot_response(user_message)
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    st.session_state.chat_history.append({"role": "user", "content": user_message})
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                    
                    conversation_id = save_conversation(
                        st.session_state.session_id, 
                        user_message, 
                        bot_response,
                        response_time
                    )
                    if conversation_id:
                        if 'conversation_ids' not in st.session_state:
                            st.session_state.conversation_ids = []
                        st.session_state.conversation_ids.append(conversation_id)
            
            # Get chat history pairs
            chat_pairs = []
            if 'chat_history' in st.session_state:
                for i in range(0, len(st.session_state.chat_history), 2):
                    if i + 1 < len(st.session_state.chat_history):
                        user_msg = st.session_state.chat_history[i]
                        bot_msg = st.session_state.chat_history[i + 1]
                        chat_pairs.append((user_msg, bot_msg))
            
            # Chat container
            chat_container = st.container()
            with chat_container:
                # Simple container with fixed height and scrolling
                st.markdown("""
                    <style>
                    .chat-container {
                        height: 600px;
                        overflow-y: auto;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 10px;
                        padding: 1rem;
                        margin-top: 1rem;
                        background-color: transparent;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                
                # Display welcome message if no messages
                if not chat_pairs:
                    st.markdown(
                        """
                        <div style="text-align: center; color: #666; padding: 20px;">
                            Start a conversation by asking a question about the MSDS program!
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Display messages in chronological order
                for i, (user_msg, bot_msg) in enumerate(chat_pairs):
                    # User message
                    st.markdown(
                        f"""
                        <div class="message user-message">
                            {user_msg["content"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Bot message
                    cleaned_bot_msg = clean_message_text(bot_msg["content"])
                    st.markdown(
                        f"""
                        <div class="message bot-message">
                            {cleaned_bot_msg}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Add feedback buttons after each bot message
                    add_feedback_buttons(i)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.sidebar.subheader("Session Management")
            st.sidebar.write(f"Session ID: {st.session_state.session_id}")
            
            if st.sidebar.button("New Session"):
                st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                st.session_state.chat_history = []
                st.rerun()
            
            st.sidebar.subheader("Example Questions")
            example_questions = [
                "What are the admission requirements for the MSDS program?",
                "How long does the MSDS program take to complete?",
                "What programming languages are taught in the program?",
                "Who are the faculty members in the MSDS program?",
                "What kind of projects do MSDS students work on?",
                "What is the tuition for the MSDS program?"
            ]
            
            for q in example_questions:
                if st.sidebar.button(q, key=f"btn_{q[:20]}"):
                    matched_question, matched_answer, similarity = find_most_similar_question(q)
                    bot_response = get_bot_response(q)
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                    save_conversation(st.session_state.session_id, q, bot_response, 0.0)

    with tab3:
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=False)
        if st.session_state.debug_mode:
            st.write("Last Query Debug Info:")
            st.write(f"Category: {st.session_state.debug_category}")
            st.write(f"Similarity Score: {st.session_state.debug_similarity:.3f}")
            st.write(f"Matched Question: {st.session_state.debug_matched_question}")
            st.write(f"Matched Answer: {st.session_state.debug_matched_answer}")

# Add after imports
def check_required_files():
    required_files = [
        "labeled_qa.csv",
        "faculty.json",
        "general_info.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.write("Please make sure all required files are in the correct location:")
        for file in missing_files:
            st.write(f"- {file}")
        st.stop()

# Add this after check_required_files()
def verify_qa_data():
    try:
        qa_df = pd.read_csv("labeled_qa.csv")
        required_columns = ['Category', 'Question', 'Answer']
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in qa_df.columns]
        if missing_columns:
            st.error(f"Missing required columns in labeled_qa.csv: {', '.join(missing_columns)}")
            st.stop()
            
        # Check if there's data
        if len(qa_df) == 0:
            st.error("labeled_qa.csv is empty")
            st.stop()
            
        # st.success(f"Successfully loaded {len(qa_df)} QA pairs") DEBUGGING
        return qa_df
    except Exception as e:
        st.error(f"Error reading labeled_qa.csv: {str(e)}")
        st.stop()

# Add this call after check_required_files()
qa_df = verify_qa_data()

# Add this near the top of your file
st.markdown(
    """
    <style>
    /* Custom scrollbar styling */
    div[data-testid="stMarkdownContainer"] {
        scrollbar-width: thin;
        scrollbar-color: #888 #f1f1f1;
    }
    
    div[data-testid="stMarkdownContainer"]::-webkit-scrollbar {
        width: 8px;
    }
    
    div[data-testid="stMarkdownContainer"]::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    div[data-testid="stMarkdownContainer"]::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    div[data-testid="stMarkdownContainer"]::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add this JavaScript for better auto-scrolling
st.markdown(
    """
    <script>
        function scrollToBottom() {
            const containers = document.getElementsByClassName('chat-container');
            if (containers.length > 0) {
                const lastContainer = containers[containers.length - 1];
                lastContainer.scrollTop = lastContainer.scrollHeight;
            }
        }
        
        // Call on load and after any content changes
        window.addEventListener('load', scrollToBottom);
        const observer = new MutationObserver(scrollToBottom);
        observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """,
    unsafe_allow_html=True
)

def clean_message_text(text):
    """Clean message text of common formatting issues"""
    return (text
        .replace("</div>", "")
        .replace("<div>", "")
        .replace("andthe", " and the ")
        .replace("andthemedianbasesalaryinternationally", " and the median base salary internationally ")
        .replace("_", "")
        .replace("  ", " ")  # Remove double spaces
        .replace("\n\n", "\n")  # Remove double line breaks
        .strip())

# Add session timeout check
def check_session_timeout(timeout_minutes=30):
    if 'last_activity' in st.session_state:
        inactive_time = datetime.now() - st.session_state.last_activity
        if inactive_time.total_seconds() > (timeout_minutes * 60):
            # Reset session
            st.session_state.chat_history = []
            st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            st.session_state.last_activity = datetime.now()
            return True
    return False

# Call this in main() before displaying chat
if check_session_timeout():
    st.info("Session timed out due to inactivity. Starting new session.")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_context_data():
    with open('context.json', 'r') as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def load_general_info():
    with open('general_info.txt', 'r') as f:
        return f.read()

def add_feedback_buttons(message_index):
    if message_index >= len(st.session_state.conversation_ids):
        return
        
    conversation_id = st.session_state.conversation_ids[message_index]
    
    # Create columns for feedback buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 4])
    
    with col1:
        if st.button("👍", key=f"thumbs_up_{message_index}"):
            update_feedback(
                conversation_id,
                "positive",
                {"reaction": "thumbs_up"}
            )
            st.success("Thank you for your feedback!")
    
    with col2:
        if st.button("👎", key=f"thumbs_down_{message_index}"):
            update_feedback(
                conversation_id,
                "negative",
                {"reaction": "thumbs_down"}
            )
            st.success("Thank you for your feedback!")
    
    with col3:
        if st.button("⚠️", key=f"report_{message_index}"):
            st.session_state[f"report_open_{message_index}"] = True
    
    # Handle detailed report submission        
    if st.session_state.get(f"report_open_{message_index}", False):
        with st.expander("Report Issue"):
            issue_type = st.selectbox(
                "Issue Type", 
                ["Incorrect Information", "Unclear Response", "Missing Information", "Other"],
                key=f"issue_type_{message_index}"
            )
            issue_description = st.text_area(
                "Description",
                key=f"issue_desc_{message_index}"
            )
            
            if st.button("Submit Report", key=f"submit_report_{message_index}"):
                update_feedback(
                    conversation_id,
                    "report",
                    {
                        "issue_type": issue_type,
                        "description": issue_description,
                        "reaction": "report"
                    }
                )
                st.success("Thank you for reporting this issue!")
                st.session_state[f"report_open_{message_index}"] = False

if __name__ == "__main__":
    main()