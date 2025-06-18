import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Optional
import os

# Configuration
API_BASE_URL = "http://localhost:8000"

def init_session_state():
    """Initialize session state variables"""
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None
    if 'agent_log' not in st.session_state:
        st.session_state.agent_log = []
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/ping", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_pdf(file):
    """Upload PDF file to the backend"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_BASE_URL}/upload-pdf/", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def ask_question(query: str) -> Optional[Dict[str, Any]]:
    """Send query to the multi-agent system"""
    try:
        payload = {"query": query}
        response = requests.post(f"{API_BASE_URL}/ask/", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending query: {str(e)}")
        return None

def send_feedback(query: str, feedback: str, original_suggestion: str) -> Optional[Dict[str, Any]]:
    """Send feedback to get revised suggestions"""
    try:
        payload = {
            "query": query,
            "feedback": feedback,
            "original_suggestion": original_suggestion
        }
        response = requests.post(f"{API_BASE_URL}/feedback/", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Feedback failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending feedback: {str(e)}")
        return None

def display_agent_log(agent_log: list):
    """Display the agent execution log"""
    if not agent_log:
        return
    
    st.subheader("Agent Execution Log")
    
    for i, step in enumerate(agent_log):
        expander_key = f"{step['step']}_{step['agent']}_{step['action']}"
        with st.expander(f"Step {step['step']}: {step['agent']} - {step['action']}"):
            result = step['result']
            
            if step['agent'] == "Retriever Agent":
                st.write(f"**Query:** {result.get('query', 'N/A')}")
                st.write(f"**Documents Retrieved:** {result.get('num_documents', 0)}")
                
                if result.get('retrieved_documents'):
                    docs = result['retrieved_documents']
                    if len(docs) > 1:
                        selected_doc_idx = st.radio(
                            "Select a document to view:",
                            options=list(range(len(docs))),
                            format_func=lambda i: f"Document {i+1}",
                            key=f"retrieved_doc_selector_{expander_key}"
                        )
                        st.text_area(
                            f"Document {selected_doc_idx+1}",
                            docs[selected_doc_idx],
                            height=100,
                            disabled=True,
                            key=f"doc_{selected_doc_idx}_retriever_{expander_key}"
                        )
                    else:
                        st.text_area(
                            "Document 1",
                            docs[0],
                            height=100,
                            disabled=True,
                            key=f"doc_1_retriever_{expander_key}"
                        )
            
            elif step['agent'] == "RFP Editor Agent":
                st.write(f"**Status:** {result.get('status', 'N/A')}")
                
                if result.get('improved_content'):
                    st.write("**Improved Content:**")
                    st.text_area(
                        "Content",
                        result['improved_content'],
                        height=200,
                        disabled=True,
                        key=f"improved_content_{expander_key}_{i}"
                    )
                
                if result.get('best_practices_applied'):
                    st.write("**Best Practices Applied:**")
                    for idx, practice in enumerate(result['best_practices_applied']):
                        st.write(f"{practice.title()}", key=f"best_practice_{idx}_{expander_key}")

def display_feedback_interface(response_data: Dict[str, Any]):
    """Display the feedback interface for user interaction"""
    st.subheader("Provide Feedback")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Accept", type="primary"):
            st.success("Accepted!")
            st.session_state.feedback_history.append({
                "action": "accepted",
                "query": response_data['query'],
                "suggestion": response_data['improvement_result']['improved_content']
            })
    
    with col2:
        if st.button("Reject"):
            st.session_state.show_reject_feedback = True
    
    with col3:
        if st.button("Edit"):
            st.session_state.show_edit_interface = True
    
    # Reject feedback interface
    if st.session_state.get('show_reject_feedback', False):
        with st.form("reject_feedback_form"):
            feedback_text = st.text_area(
                "Why are you rejecting this suggestion?",
                placeholder="Please provide specific feedback on what needs to be improved...",
                key="reject_feedback_text"
            )
            
            if st.form_submit_button("Submit Feedback"):
                if feedback_text.strip():
                    # Send feedback to get revised suggestion
                    revised_response = send_feedback(
                        response_data['query'],
                        feedback_text,
                        response_data['improvement_result']['improved_content']
                    )
                    
                    if revised_response:
                        st.session_state.current_response = revised_response
                        st.session_state.feedback_history.append({
                            "action": "rejected",
                            "query": response_data['query'],
                            "original_suggestion": response_data['improvement_result']['improved_content'],
                            "feedback": feedback_text,
                            "revised_suggestion": revised_response['revision_result']['improved_content']
                        })
                        st.rerun()
                else:
                    st.error("Please provide feedback text.")
    
    # Edit interface
    if st.session_state.get('show_edit_interface', False):
        with st.form("edit_interface_form"):
            st.write("**Original Suggestion:**")
            st.text_area("Original", response_data['improvement_result']['improved_content'], height=150, disabled=True, key="original_suggestion")
            
            user_edit = st.text_area(
                "Your Edited Version:",
                value=response_data['improvement_result']['improved_content'],
                height=200,
                key="user_edit"
            )
            
            if st.form_submit_button("Save Edit"):
                if user_edit.strip():
                    st.session_state.feedback_history.append({
                        "action": "edited",
                        "query": response_data['query'],
                        "original_suggestion": response_data['improvement_result']['improved_content'],
                        "user_edit": user_edit
                    })
                    st.success("Your edit has been saved!")
                    st.rerun()
                else:
                    st.error("Please provide your edited version.")

def display_feedback_history():
    """Display the history of user feedback and actions"""
    if not st.session_state.feedback_history:
        return
    
    st.subheader("Feedback History")
    
    for i, feedback in enumerate(st.session_state.feedback_history, 1):
        with st.expander(f"Feedback #{i}: {feedback['action'].title()}"):
            st.write(f"**Query:** {feedback['query']}")
            
            if feedback['action'] == 'accepted':
                st.write("Accepted")
                st.write("**Accepted Suggestion:**")
                st.text_area(f"Suggestion {i}", feedback['suggestion'], height=150, disabled=True, key=f"accepted_suggestion_{i}")
            
            elif feedback['action'] == 'rejected':
                st.write("Rejected")
                st.write("**Original Suggestion:**")
                st.text_area(f"Original {i}", feedback['original_suggestion'], height=100, disabled=True, key=f"original_suggestion_{i}")
                st.write("**Feedback:**")
                st.text_area(f"Feedback {i}", feedback['feedback'], height=100, disabled=True, key=f"feedback_text_{i}")
                st.write("**Revised Suggestion:**")
                st.text_area(f"Revised {i}", feedback['revised_suggestion'], height=150, disabled=True, key=f"revised_suggestion_{i}")
            
            elif feedback['action'] == 'edited':
                st.write("Edited")
                st.write("**Original Suggestion:**")
                st.text_area(f"Original {i}", feedback['original_suggestion'], height=100, disabled=True, key=f"original_suggestion_{i}")
                st.write("**Your Edit:**")
                st.text_area(f"Edit {i}", feedback['user_edit'], height=150, disabled=True, key=f"user_edit_{i}")

def main():
    st.set_page_config(
        page_title="Multi-Agent RFP Assistant",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Header
    st.title("Multi-Agent RFP Assistant")
    st.markdown("""
    This system uses two specialized agents to help you review and improve RFP content:
    - Retriever Agent: Finds relevant documents from your knowledge base
    - RFP Editor Agent: Analyzes content and suggests improvements based on RFP best practices
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Health Check
        if check_api_health():
            st.info("API Connected")
        else:
            st.error("API Not Connected")
            st.info("Please start the FastAPI backend server")
            return
        
        # File Upload
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload RFP documents for analysis"
        )
        
        if uploaded_file is not None:
            if st.button("Upload PDF"):
                with st.spinner("Uploading and processing PDF..."):
                    result = upload_pdf(uploaded_file)
                    if result:
                        st.info(f"{uploaded_file.name} uploaded successfully!")
                        st.info(f"Task ID: {result.get('task_id', 'N/A')}")
        
        # Configuration
        st.subheader("Settings")
        if st.button("View API Config"):
            try:
                config_response = requests.get(f"{API_BASE_URL}/config")
                if config_response.status_code == 200:
                    config = config_response.json()
                    st.json(config)
            except Exception as e:
                st.error(f"Error fetching config: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask Your Question")
        
        # Query input
        query = st.text_area(
            "Enter your RFP-related question or request for improvement:",
            placeholder="e.g., 'Help me improve the project scope section' or 'Review this technical requirements section'",
            height=100,
            key="query_input"
        )
        
        if st.button("Process with Multi-Agent System", type="primary"):
            if query.strip():
                with st.spinner("Processing with multi-agent system..."):
                    response = ask_question(query)
                    if response:
                        st.session_state.current_query = query
                        st.session_state.current_response = response
                        st.session_state.agent_log = response.get('agent_log', [])
                        st.rerun()
            else:
                st.error("Please enter a query.")
        
        # Display current response
        if st.session_state.current_response:
            st.subheader("Results")
            
            response_data = st.session_state.current_response
            
            # Display retrieval results
            retrieval = response_data['retrieval_result']
            st.write(f"Retriever Agent Results:")
            st.info(f"Found {retrieval.get('num_documents', 0)} relevant documents")
            
            if retrieval.get('retrieved_documents'):
                docs = retrieval['retrieved_documents']
                if len(docs) > 1:
                    selected_doc_idx = st.radio(
                        "Select a document to view:",
                        options=list(range(len(docs))),
                        format_func=lambda i: f"Document {i+1}",
                        key="retrieved_doc_selector_main"
                    )
                    st.text_area(
                        f"Content {selected_doc_idx+1}",
                        docs[selected_doc_idx],
                        height=100,
                        disabled=True,
                        key=f"doc_{selected_doc_idx}_retriever_main"
                    )
                else:
                    st.text_area(
                        "Content 1",
                        docs[0],
                        height=100,
                        disabled=True,
                        key="doc_1_retriever_main"
                    )
            
            # Display improvement results
            improvement = response_data['improvement_result']
            st.write(f"RFP Editor Agent Results:")
            
            if improvement.get('improved_content'):
                st.text_area(
                    "Improved Content",
                    improvement['improved_content'],
                    height=300,
                    disabled=True,
                    key="improved_content_rfp_editor_main"
                )
                
                if improvement.get('best_practices_applied'):
                    st.write("Applied Best Practices:")
                    for idx, practice in enumerate(improvement['best_practices_applied']):
                        st.write(f"{practice.title()}", key=f"best_practice_{idx}_main")
            
            # Feedback interface
            display_feedback_interface(response_data)
    
    with col2:
        st.header("Quick Actions")
        
        # Clear session
        if st.button("Clear Session"):
            st.session_state.current_query = ""
            st.session_state.current_response = None
            st.session_state.agent_log = []
            st.session_state.feedback_history = []
            st.rerun()
        
        # Display agent log
        if st.session_state.agent_log:
            display_agent_log(st.session_state.agent_log)
        
        # Display feedback history
        display_feedback_history()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Multi-Agent RFP Assistant | Powered by Ollama & ChromaDB</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 