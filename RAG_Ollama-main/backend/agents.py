import ollama
import logging
from typing import List, Dict, Any, Tuple
from .config import Config
import re
from rag_pipeline import get_all_paragraph_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrieverAgent:
    """Agent A: Responsible for retrieving relevant documents from ChromaDB"""
    
    def __init__(self, query_vector_db_func):
        self.query_vector_db = query_vector_db_func
        self.name = "Retriever Agent"
    
    def retrieve(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Retrieve paragraphs containing exact matches for the keyword/phrase in the uploaded documents.
        """
        try:
            if top_k is None:
                top_k = Config.TOP_K_RESULTS
            logger.info(f"{self.name}: Passing query to LLM for retrieval and answer generation: '{query}'")
            all_chunks = get_all_paragraph_chunks()
            logger.info(f"{self.name}: Retrieved {len(all_chunks)} total paragraph chunks from DB")
            # Concatenate all paragraphs as context
            context = "\n\n".join([chunk['text'] for chunk in all_chunks if isinstance(chunk, dict) and 'text' in chunk and chunk['text']])
            llm_answer = ""
            if context:
                prompt = f"You are an expert assistant. Use the following document context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nIf the answer is not in the context, say so."
                response = ollama.chat(
                    model=Config.get_ollama_model(),
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": Config.TEMPERATURE}
                )
                llm_answer = response['message']['content']
            else:
                llm_answer = "No document context is available to answer the question."
            return {
                "query": query,
                "context": context,
                "llm_answer": llm_answer,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"{self.name}: Error during retrieval: {e}")
            return {
                "query": query,
                "context": "",
                "llm_answer": "",
                "status": "error",
                "error": str(e)
            }

class RFPEditorAgent:
    """Agent B: Responsible for analyzing and improving RFP content"""
    
    def __init__(self):
        self.name = "RFP Editor Agent"
        self.rfp_best_practices = self._get_rfp_best_practices()
    
    def _get_rfp_best_practices(self) -> str:
        """Get the RFP best practices checklist"""
        return """
        RFP BEST PRACTICES CHECKLIST:
        
        ✅ CLARITY & SCOPE:
        - Clear, unambiguous objectives and goals
        - Well-defined scope of work
        - Specific deliverables and outcomes
        
        ✅ MEASURABLE OUTCOMES:
        - Quantifiable success metrics
        - Key Performance Indicators (KPIs)
        - Evaluation criteria
        
        ✅ STAKEHOLDER NEEDS:
        - Identified stakeholders and their requirements
        - User needs and pain points
        - Business requirements
        
        ✅ VENDOR RESPONSIBILITIES:
        - Clear vendor roles and responsibilities
        - Required qualifications and experience
        - Performance expectations
        
        ✅ TIMELINE & BUDGET:
        - Realistic project timeline
        - Budget constraints and payment terms
        - Milestone definitions
        
        ✅ EXAMPLES & CONTEXT:
        - Real-world examples and use cases
        - Industry-specific terminology
        - Context for requirements
        
        ✅ TECHNICAL SPECIFICATIONS:
        - Detailed technical requirements
        - Integration requirements
        - Security and compliance needs
        """
    
    def analyze_and_improve(self, query: str, context: str, original_response: str = None) -> Dict[str, Any]:
        """
        Analyze the content and provide improvement suggestions
        
        Args:
            query: The user's original question
            context: Retrieved context from Agent A
            original_response: Previous response if this is a revision
            
        Returns:
            Dictionary containing analysis and suggestions
        """
        try:
            logger.info(f"{self.name}: Analyzing content for improvement")
            
            # Create the analysis prompt
            analysis_prompt = self._create_analysis_prompt(query, context, original_response)
            
            # Get response from Ollama
            response = ollama.chat(
                model=Config.get_ollama_model(),
                messages=[{"role": "user", "content": analysis_prompt}],
                options={"temperature": Config.TEMPERATURE}
            )
            
            improved_content = response['message']['content']
            
            logger.info(f"{self.name}: Generated improvement suggestions")
            
            return {
                "original_query": query,
                "context_used": context,
                "improved_content": improved_content,
                "best_practices_applied": self._extract_applied_practices(improved_content),
                "status": "success",
                "agent_name": self.name
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error during analysis: {e}")
            return {
                "original_query": query,
                "context_used": context,
                "improved_content": "",
                "best_practices_applied": [],
                "status": "error",
                "error": str(e),
                "agent_name": self.name
            }
    
    def _create_analysis_prompt(self, query: str, context: str, original_response: str = None) -> str:
        """Create the analysis prompt with intelligent response logic"""
        
        # Check if the query is asking about content retrieval
        retrieval_keywords = ['is there', 'does it contain', 'does the document', 'is mentioned', 'can you find', 'look for', 'search for', 'find', 'locate', 'where is', 'what does it say about']
        is_retrieval_query = any(keyword in query.lower() for keyword in retrieval_keywords)
        
        if is_retrieval_query and context.strip():
            # User is asking if something exists in the PDF - provide relevant content
            base_prompt = f"""
            You are an expert assistant analyzing a PDF document. The user is asking if specific content exists in the document.

            USER QUERY: {query}

            RELEVANT CONTENT FROM DOCUMENT:
            {context}

            INSTRUCTIONS:
            - If the relevant content is found, clearly state what was found and provide the specific information
            - If the content is not found, clearly state that it was not found in the document
            - Always be specific about what was found or not found
            - Use the exact content from the document when possible
            - If the content is partially relevant, explain what was found and what might be missing

            Please provide a clear, direct answer based on the document content.
            """
        elif is_retrieval_query and not context.strip():
            # User is asking for content but nothing was found
            base_prompt = f"""
            You are an expert assistant analyzing a PDF document. The user is asking if specific content exists in the document.

            USER QUERY: {query}

            RESULT: No relevant content was found in the document for this query.

            INSTRUCTIONS:
            - Clearly state that the requested content was not found in the document
            - Suggest what the user might be looking for or alternative ways to phrase their question
            - Be helpful and informative even when content is not found

            Please provide a helpful response explaining that the content was not found.
            """
        else:
            # User is not asking for retrieval - answer the question directly
            base_prompt = f"""
            You are an expert assistant. The user has asked a question that does not require searching through document content.

            USER QUERY: {query}

            INSTRUCTIONS:
            - Answer the question directly based on your knowledge
            - Do not reference any document content unless specifically relevant
            - Provide helpful, accurate, and informative responses
            - If the question is about RFP best practices, use your expertise to provide guidance

            Please provide a direct answer to the user's question.
            """
        
        if original_response:
            base_prompt += f"""
            
            PREVIOUS RESPONSE (if this is a revision):
            {original_response}
            
            Please provide an improved version that addresses any feedback or concerns.
            """
        
        return base_prompt
    
    def _extract_applied_practices(self, content: str) -> List[str]:
        """Extract which best practices were applied in the response"""
        applied_practices = []
        
        practice_keywords = {
            "clarity": ["clear", "unambiguous", "specific", "well-defined"],
            "measurable": ["measurable", "quantifiable", "kpi", "metrics"],
            "stakeholders": ["stakeholder", "user needs", "requirements"],
            "responsibilities": ["responsibility", "role", "duties"],
            "timeline": ["timeline", "schedule", "milestone", "deadline"],
            "budget": ["budget", "cost", "payment", "financial"],
            "examples": ["example", "use case", "scenario", "instance"]
        }
        
        content_lower = content.lower()
        for practice, keywords in practice_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                applied_practices.append(practice)
        
        return applied_practices
    
    def rephrase_with_feedback(self, query: str, context: str, feedback: str, original_suggestion: str) -> Dict[str, Any]:
        """
        Rephrase the suggestion based on user feedback
        
        Args:
            query: Original user query
            context: Retrieved context
            feedback: User feedback (rejection reason)
            original_suggestion: The suggestion that was rejected
            
        Returns:
            Dictionary containing the rephrased suggestion
        """
        try:
            logger.info(f"{self.name}: Rephrasing based on user feedback")
            
            # Check if the query is asking about content retrieval
            retrieval_keywords = ['is there', 'does it contain', 'does the document', 'is mentioned', 'can you find', 'look for', 'search for', 'find', 'locate', 'where is', 'what does it say about']
            is_retrieval_query = any(keyword in query.lower() for keyword in retrieval_keywords)
            
            if is_retrieval_query:
                rephrase_prompt = f"""
                The user rejected your previous suggestion. Please provide an improved version.
                
                ORIGINAL QUERY: {query}
                CONTEXT: {context}
                YOUR PREVIOUS SUGGESTION: {original_suggestion}
                USER FEEDBACK: {feedback}
                
                INSTRUCTIONS:
                - If the relevant content is found, clearly state what was found and provide the specific information
                - If the content is not found, clearly state that it was not found in the document
                - Always be specific about what was found or not found
                - Use the exact content from the document when possible
                - Address the user's feedback in your response
                
                Please provide a new suggestion that addresses the user's feedback.
                """
            else:
                rephrase_prompt = f"""
                The user rejected your previous suggestion. Please provide an improved version.
                
                ORIGINAL QUERY: {query}
                YOUR PREVIOUS SUGGESTION: {original_suggestion}
                USER FEEDBACK: {feedback}
                
                INSTRUCTIONS:
                - Answer the question directly based on your knowledge
                - Do not reference any document content unless specifically relevant
                - Provide helpful, accurate, and informative responses
                - Address the user's feedback in your response
                
                Please provide a new suggestion that addresses the user's feedback.
                """
            
            response = ollama.chat(
                model=Config.get_ollama_model(),
                messages=[{"role": "user", "content": rephrase_prompt}],
                options={"temperature": Config.TEMPERATURE}
            )
            
            rephrased_content = response['message']['content']
            
            return {
                "original_query": query,
                "context_used": context,
                "improved_content": rephrased_content,
                "feedback_addressed": feedback,
                "status": "success",
                "agent_name": self.name
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error during rephrasing: {e}")
            return {
                "original_query": query,
                "context_used": context,
                "improved_content": "",
                "feedback_addressed": feedback,
                "status": "error",
                "error": str(e),
                "agent_name": self.name
            }

class HelpingAgent:
    """Agent that answers RFP-related questions using both general RFP knowledge and indexed PDFs."""
    def __init__(self, query_vector_db_func):
        self.query_vector_db = query_vector_db_func
        self.name = "Helping Agent"

    def answer(self, query: str) -> str:
        try:
            top_k = Config.TOP_K_RESULTS
            context_chunks = self.query_vector_db(query, n_results=top_k)
            logger.info(f"{self.name} context_chunks: {context_chunks}")
            if not isinstance(context_chunks, list):
                context_chunks = []
            filtered_chunks = []
            if context_chunks:
                for chunk in context_chunks:
                    if isinstance(chunk, dict) and 'text' in chunk and chunk['text'] is not None:
                        filtered_chunks.append(chunk)
            context = "\n".join([chunk['text'] for chunk in filtered_chunks]) if filtered_chunks else ""

            prompt = f"""
            You are an expert in writing, reviewing, and consulting on Requests for Proposal (RFPs). Answer the user's question with clear, accurate, and practical advice. Use both your general RFP knowledge and the provided document context below. If the context is relevant, cite it in your answer. If not, answer from your expertise.

            USER QUESTION: {query}

            DOCUMENT CONTEXT:
            {context}
            """
            response = ollama.chat(
                model=Config.get_ollama_model(),
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": Config.TEMPERATURE}
            )
            if not response or not isinstance(response, dict):
                logger.error(f"{self.name}: ollama.chat returned None or invalid response: {response}")
                return "Error: LLM did not return a valid response."
            message = response.get('message')
            if not message or not isinstance(message, dict):
                logger.error(f"{self.name}: ollama.chat response missing 'message': {response}")
                return "Error: LLM response missing message content."
            content = message.get('content')
            if not content:
                logger.error(f"{self.name}: ollama.chat message missing 'content': {response}")
                return "Error: LLM response missing content."
            # Remove any preamble or 'thoughts' before the answer
            # Heuristic: take content after the first double linebreak or 'Answer:'
            split_content = re.split(r"\n\n|Answer:|A:|\n[Aa]nswer", content, maxsplit=1)
            if len(split_content) > 1:
                answer = split_content[1].strip()
            else:
                answer = content.strip()
            return answer
        except Exception as e:
            logger.error(f"{self.name}: Error in answer: {e}")
            return f"Error: {str(e)}"

class MultiAgentRFPAssistant:
    """Main coordinator for the multi-agent RFP review system"""
    
    def __init__(self, query_vector_db_func):
        self.retriever_agent = RetrieverAgent(query_vector_db_func)
        self.rfp_editor_agent = RFPEditorAgent()
        self.agent_log = []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the multi-agent pipeline
        
        Args:
            query: User's question or request
            
        Returns:
            Dictionary containing results from pdf
        """
        logger.info("MultiAgentRFPAssistant: Starting query processing")
        
        # Always use paragraph containment logic for retrieval
        retrieval_result = self.retriever_agent.retrieve(query)
        self.agent_log.append({
            "step": 1,
            "agent": "Retriever Agent",
            "action": "Document retrieval",
            "result": retrieval_result
        })
        if retrieval_result["status"] == "error":
            return {
                "status": "error",
                "error": "Failed to retrieve documents",
                "agent_log": self.agent_log
            }
        # Step 2: Agent B - Analyze and improve content
        if retrieval_result["context"]:
            improvement_result = self.rfp_editor_agent.analyze_and_improve(
                query, 
                retrieval_result["context"]
            )
        else:
            improvement_result = self.rfp_editor_agent.analyze_and_improve(
                query, 
                ""  # Empty context for no results found
            )
        self.agent_log.append({
            "step": 2,
            "agent": "RFP Editor Agent",
            "action": "Content analysis and improvement",
            "result": improvement_result
        })
        return {
            "status": "success",
            "query": query,
            "retrieval_result": retrieval_result,
            "improvement_result": improvement_result,
            "agent_log": self.agent_log
        }
    
    def handle_feedback(self, query: str, feedback: str, original_suggestion: str) -> Dict[str, Any]:
        """
        Handle user feedback and provide an improved response
        
        Args:
            query: Original user query
            feedback: User feedback
            original_suggestion: The original suggestion that received feedback
            
        Returns:
            Dictionary containing the improved response
        """
        try:
            logger.info(f"{self.name}: Handling user feedback")
            
            # Get context if needed for retrieval queries
            retrieval_keywords = ['is there', 'does it contain', 'does the document', 'is mentioned', 'can you find', 'look for', 'search for', 'find', 'locate', 'where is', 'what does it say about']
            is_retrieval_query = any(keyword in query.lower() for keyword in retrieval_keywords)
            
            if is_retrieval_query:
                # For retrieval queries, we need to get context first
                context_result = self.retriever_agent.retrieve(query)
                context = context_result.get('context', '')
                return self.rfp_editor_agent.rephrase_with_feedback(query, context, feedback, original_suggestion)
            else:
                # For non-retrieval queries, answer directly without context
                return self.rfp_editor_agent.rephrase_with_feedback(query, '', feedback, original_suggestion)
                
        except Exception as e:
            logger.error(f"{self.name}: Error handling feedback: {e}")
            return {
                "original_query": query,
                "improved_content": "",
                "feedback_addressed": feedback,
                "status": "error",
                "error": str(e),
                "agent_name": self.name
            } 