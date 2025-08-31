#!/usr/bin/env python3
"""
Standalone Q&A System with RAG and Llama 3.1

This module provides a simple interface for answering questions using
retrieval-augmented generation with Llama 3.1 model.
"""

import sys
import os
import json
from typing import List, Dict, Any
import requests
from retriever import EmbeddingRetriever

class QASystem:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.1"):
        """
        Initialize the Q&A system.
        
        Args:
            ollama_url: URL of the Ollama server
            model: Name of the model to use
        """
        self.ollama_url = ollama_url
        self.model = model
        self.retriever = EmbeddingRetriever()
        
    def create_professional_prompt(self, question: str, context: List[Dict[str, Any]]) -> str:
        """
        Create a professional prompt for Llama 3.1 with retrieved context.
        
        Args:
            question: User's question
            context: Retrieved document chunks with metadata
            
        Returns:
            Formatted prompt string
        """
        # Format context documents
        context_text = ""
        for i, doc in enumerate(context, 1):
            context_text += f"\n[Document {i}]\n"
            context_text += f"Content: {doc['content']}\n"
            if 'source' in doc:
                context_text += f"Source: {doc['source']}\n"
            context_text += "-" * 50 + "\n"
        
        # Detect question language
        question_lower = question.lower()
        french_indicators = ['quoi', 'que', 'comment', 'pourquoi', 'où', 'quand', 'qui', 'quel', 'quelle', 
                           "qu'est-ce", "qu'est", 'est-ce', 'une', 'des', 'les', 'dans', 'avec', 'pour', 
                           'assurance', 'prime', 'contrat', 'garantie']
        is_french = any(word in question_lower for word in french_indicators)
        is_arabic = any(char in question for char in 'أإآةتثجحخدذرزسشصضطظعغفقكلمنهوىي')
        
        if is_arabic:
            prompt = f"""أنت مساعد ذكي متخصص في التأمين لشركة BH للتأمين. يجب أن تقدم إجابات دقيقة ومختصرة باللغة العربية.

**التعليمات:**
1. أجب على السؤال باستخدام المعلومات المتوفرة في الوثائق أدناه فقط
2. إذا لم تحتوي الوثائق على معلومات كافية، اذكر ذلك بوضوح
3. قدم إجابة دقيقة ومختصرة (لا تزيد عن 3-4 جمل)
4. اذكر المصادر عند الضرورة
5. حافظ على طابع مهني ومفيد

**وثائق السياق:**{context_text}

**السؤال:** {question}

**الإجابة:**"""
        elif is_french:
            prompt = f"""Vous êtes un assistant IA spécialisé dans l'assurance pour BH Assurance. Vous devez fournir des réponses précises et concises en français.

**INSTRUCTIONS:**
1. Répondez à la question en utilisant UNIQUEMENT les informations fournies dans les documents ci-dessous
2. Si les documents ne contiennent pas suffisamment d'informations, indiquez-le clairement
3. Fournissez une réponse précise et concise (maximum 3-4 phrases)
4. Citez les sources si nécessaire
5. Maintenez un ton professionnel et utile

**DOCUMENTS DE CONTEXTE:**{context_text}

**QUESTION:** {question}

**RÉPONSE:**"""
        else:
            prompt = f"""You are a professional AI assistant specialized in insurance for BH Insurance. You must provide accurate and concise responses.

**INSTRUCTIONS:**
1. Answer the question using ONLY the information provided in the context documents below
2. If the context doesn't contain enough information, clearly state this limitation
3. Provide a precise and concise response (maximum 3-4 sentences)
4. Cite sources when necessary
5. Maintain a professional and helpful tone

**CONTEXT DOCUMENTS:**{context_text}

**QUESTION:** {question}

**ANSWER:**"""
        
        return prompt
    
    def call_llama(self, prompt: str) -> str:
        """
        Call Llama 3.1 model via Ollama API.
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            Generated response from the model
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2048,
                        "stop": ["\n\n**QUESTION:**", "\n\n**CONTEXT DOCUMENTS:**"]
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: Failed to get response from model (Status: {response.status_code})"
                
        except requests.exceptions.RequestException as e:
            return f"Error: Failed to connect to Ollama server - {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error - {str(e)}"
    
    def answer_question(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using RAG approach.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        try:
            # Retrieve relevant documents
            print(f"🔍 Searching for relevant information...")
            search_results = self.retriever.search(question, top_k=k)
            
            if not search_results:
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "metadata": {
                        "retrieved_docs": 0,
                        "model": self.model,
                        "status": "no_results"
                    }
                }
            
            print(f"📚 Found {len(search_results)} relevant documents")
            
            # Convert SearchResult objects to dictionaries for prompt creation
            context_docs = []
            for result in search_results:
                context_docs.append({
                    'content': result.text,
                    'source': result.filename,
                    'score': result.score
                })
            
            # Create professional prompt
            prompt = self.create_professional_prompt(question, context_docs)
            
            # Generate answer using Llama 3.1
            print(f"🤖 Generating answer with {self.model}...")
            answer = self.call_llama(prompt)
            
            # Extract sources
            sources = []
            for result in search_results:
                source_info = {
                    "content": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                    "score": result.score,
                    "source": result.filename
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "retrieved_docs": len(search_results),
                    "model": self.model,
                    "status": "success"
                }
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "metadata": {
                    "retrieved_docs": 0,
                    "model": self.model,
                    "status": "error",
                    "error": str(e)
                }
            }

def main():
    """
    Main function for command-line usage.
    """
    print("🚀 Q&A System with RAG and Llama 3.1")
    print("=" * 40)
    
    # Initialize Q&A system
    qa_system = QASystem()
    
    if len(sys.argv) > 1:
        # Question provided as command line argument
        question = " ".join(sys.argv[1:])
        print(f"\n❓ Question: {question}\n")
        
        result = qa_system.answer_question(question)
        
        print("\n💡 Answer:")
        print("-" * 20)
        print(result['answer'])
        
        if result['sources']:
            print("\n📖 Sources:")
            print("-" * 20)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['content']} (Score: {source['score']:.3f})")
                if 'source' in source:
                    print(f"   Source: {source['source']}")
                print()
        
        print(f"\n📊 Metadata: {json.dumps(result['metadata'], indent=2)}")
        
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("❓ Enter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print()
                result = qa_system.answer_question(question)
                
                print("\n💡 Answer:")
                print("-" * 20)
                print(result['answer'])
                
                if result['sources']:
                    print("\n📖 Sources:")
                    print("-" * 20)
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['content']} (Score: {source['score']:.3f})")
                        if 'source' in source:
                            print(f"   Source: {source['source']}")
                        print()
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

if __name__ == "__main__":
    main()