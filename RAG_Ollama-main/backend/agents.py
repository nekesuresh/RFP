import ollama
import logging
from typing import List, Dict, Any, Tuple
from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrieverAgent:
    """Agent A: Responsible for retrieving relevant documents from ChromaDB"""
    
    def __init__(self, query_vector_db_func):
        self.query_vector_db = query_vector_db_func
        self.name = "Retriever Agent"
    
    def retrieve(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Retrieve relevant documents for the given query
        
        Args:
            query: The user's question or request
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        try:
            if top_k is None:
                top_k = Config.TOP_K_RESULTS
                
            logger.info(f"{self.name}: Retrieving documents for query: {query}")
            
            # Get relevant documents from vector database
            context_docs = self.query_vector_db(query, top_k)
            context = "\n".join(context_docs) if context_docs else ""
            
            logger.info(f"{self.name}: Retrieved {len(context_docs)} documents")
            
            return {
                "query": query,
                "retrieved_documents": context_docs,
                "context": context,
                "num_documents": len(context_docs),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error during retrieval: {e}")
            return {
                "query": query,
                "retrieved_documents": [],
                "context": "",
                "num_documents": 0,
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
        """Create the analysis prompt with RFP best practices"""
        
        base_prompt = f"""
        You are an expert RFP (Request for Proposal) editor and consultant. Your task is to analyze the given content and provide improvements based on RFP best practices.

        {self.rfp_best_practices}

        USER QUERY: {query}

        CONTEXT FROM DOCUMENTS:
        {context}

        """
        
        if original_response:
            base_prompt += f"""
            PREVIOUS RESPONSE (if this is a revision):
            {original_response}
            
            Please provide an improved version that addresses any feedback or concerns.
            """
        else:
            base_prompt += """
            Please analyze the content and provide:
            1. An improved version that follows RFP best practices
            2. Specific suggestions for enhancement
            3. Areas that need clarification or additional detail
            4. Recommendations for better structure and clarity
            
            Format your response as:
            ---IMPROVED CONTENT---
            [Your improved version here]
            
            ---SUGGESTIONS---
            [List of specific improvements made]
            
            ---AREAS FOR CLARIFICATION---
            [What needs more detail or clarification]
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
            
            rephrase_prompt = f"""
            The user rejected your previous suggestion. Please provide an improved version.
            
            ORIGINAL QUERY: {query}
            CONTEXT: {context}
            YOUR PREVIOUS SUGGESTION: {original_suggestion}
            USER FEEDBACK: {feedback}
            
            {self.rfp_best_practices}
            
            Please provide a new suggestion that addresses the user's feedback while maintaining RFP best practices.
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
            Dictionary containing results from both agents
        """
        logger.info("MultiAgentRFPAssistant: Starting query processing")
        
        # Step 1: Agent A - Retrieve relevant documents
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
            improvement_result = {
                "original_query": query,
                "context_used": "",
                "improved_content": "No relevant documents found to analyze.",
                "best_practices_applied": [],
                "status": "no_context",
                "agent_name": "RFP Editor Agent"
            }
        
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
        Handle user feedback and generate revised suggestions
        
        Args:
            query: Original user query
            feedback: User feedback (rejection reason)
            original_suggestion: The suggestion that was rejected
            
        Returns:
            Dictionary containing the revised suggestion
        """
        logger.info("MultiAgentRFPAssistant: Handling user feedback")
        
        # Get the original context from the log
        original_context = ""
        if self.agent_log:
            retrieval_result = self.agent_log[0]["result"]
            original_context = retrieval_result.get("context", "")
        
        # Generate revised suggestion
        revision_result = self.rfp_editor_agent.rephrase_with_feedback(
            query, original_context, feedback, original_suggestion
        )
        
        self.agent_log.append({
            "step": len(self.agent_log) + 1,
            "agent": "RFP Editor Agent",
            "action": "Feedback-based revision",
            "result": revision_result
        })
        
        return {
            "status": "success",
            "query": query,
            "feedback": feedback,
            "revision_result": revision_result,
            "agent_log": self.agent_log
        } 