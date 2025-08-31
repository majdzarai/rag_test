import os
import json
import uuid
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx
from sentence_transformers import SentenceTransformer
import faiss

from retriever import EmbeddingRetriever
from config_emb import get_index_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global state
retriever_instance: Optional[EmbeddingRetriever] = None
metadata_info: Optional[Dict[str, Any]] = None

# Environment configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama')
LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.1:latest')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    k: int = Field(default=5, description="Number of documents to retrieve")
    mmr: bool = Field(default=True, description="Use MMR re-ranking")
    lambda_: float = Field(default=0.5, alias="lambda", description="MMR lambda parameter")
    temperature: float = Field(default=0.2, description="LLM temperature")
    debug: bool = Field(default=False, description="Include debug information")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(default=5, description="Number of documents to retrieve")
    mmr: bool = Field(default=True, description="Use MMR re-ranking")
    lambda_: float = Field(default=0.5, alias="lambda", description="MMR lambda parameter")

class Hit(BaseModel):
    id: str
    score: float
    text: str
    filename: str
    section: str
    category: str

class Citation(BaseModel):
    id: str
    filename: str
    section: str
    score: float

class DoneEvent(BaseModel):
    citations: List[Citation]
    used_k: int
    provider: str
    model: str
    debug_chunks: Optional[List[Hit]] = None

class SearchResponse(BaseModel):
    hits: List[Hit]
    query: str
    used_k: int
    mmr_enabled: bool

class ChatResponse(BaseModel):
    content: str
    citations: List[Citation]
    used_k: int
    provider: str
    model: str
    debug_chunks: Optional[List[Hit]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"

class MetaResponse(BaseModel):
    build_info: Dict[str, Any]
    index_stats: Dict[str, Any]

# Language detection
def detect_language(text: str) -> str:
    """Simple heuristic language detection for Arabic vs Latin scripts"""
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return 'en'
    
    arabic_ratio = arabic_chars / total_chars
    if arabic_ratio > 0.3:
        return 'ar'
    elif any(char in 'àâäéèêëïîôöùûüÿçñ' for char in text.lower()):
        return 'fr'
    else:
        return 'en'

# LLM Client
class LLMClient:
    def __init__(self, provider: str = LLM_PROVIDER, model: str = LLM_MODEL):
        self.provider = provider
        self.model = model
        self.ollama_url = f"{OLLAMA_BASE_URL}/api/chat"
        
    def get_system_prompt(self, language: str) -> str:
        """Get system prompt based on detected language"""
        if language == 'ar':
            return """أنت مساعد تأمين مفيد لشركة BH Assurance.
أجب باللغة العربية باستخدام السياق المقدم فقط.
إذا لم تكن الإجابة في السياق، قل أنك لا تملك هذه المعلومات.
استشهد بالمصادر باستخدام [اسم الملف • القسم]."""
        elif language == 'fr':
            return """Vous êtes un assistant d'assurance utile pour BH Assurance.
Répondez en français en utilisant uniquement le CONTEXTE fourni.
Si la réponse n'est pas dans le contexte, dites que vous n'avez pas cette information.
Citez les sources avec [nom_fichier • section]."""
        else:
            return """You are a helpful insurance assistant for BH Assurance.
Answer in the user's language using only the provided CONTEXT.
If the answer isn't in context, say you don't have that info.
Cite sources with [filename • section]."""
    
    def build_prompt(self, messages: List[ChatMessage], context: str, language: str) -> List[Dict[str, str]]:
        """Build the final prompt with system message and context"""
        system_prompt = self.get_system_prompt(language)
        
        prompt_messages = [
            {"role": "system", "content": f"{system_prompt}\n\nCONTEXT:\n{context}"}
        ]
        
        # Add conversation history
        for msg in messages:
            prompt_messages.append({"role": msg.role, "content": msg.content})
        
        return prompt_messages
    
    async def stream_chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> AsyncGenerator[str, None]:
        """Stream chat response from LLM"""
        if self.provider == 'ollama':
            async for chunk in self._stream_ollama(messages, temperature):
                yield chunk
        elif self.provider == 'openai':
            async for chunk in self._stream_openai(messages, temperature):
                yield chunk
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    async def _stream_ollama(self, messages: List[Dict[str, str]], temperature: float) -> AsyncGenerator[str, None]:
        """Stream from Ollama API"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": 2048
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream('POST', self.ollama_url, json=payload) as response:
                    if response.status_code != 200:
                        raise HTTPException(status_code=500, detail=f"Ollama API error: {response.status_code}")
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if 'message' in data and 'content' in data['message']:
                                    content = data['message']['content']
                                    if content:
                                        yield content
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            yield f"Error: Unable to generate response. {str(e)}"
    
    async def _stream_openai(self, messages: List[Dict[str, str]], temperature: float) -> AsyncGenerator[str, None]:
        """Stream from OpenAI API (placeholder implementation)"""
        # This would require openai library and proper implementation
        yield "OpenAI streaming not implemented yet. Please use Ollama."

# Startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global retriever_instance, metadata_info
    
    logger.info("Starting BH Assurance RAG API...")
    
    try:
        # Load metadata
        index_dir = get_index_dir()
        metadata_path = os.path.join(index_dir, 'metadata.parquet')
        
        if os.path.exists(metadata_path):
            metadata_df = pd.read_parquet(metadata_path)
            metadata_info = {
                'total_chunks': len(metadata_df),
                'model_name': metadata_df['model_name'].iloc[0] if 'model_name' in metadata_df.columns else 'sentence-transformers/all-MiniLM-L6-v2',
                'embedding_dim': metadata_df['embedding_dim'].iloc[0] if 'embedding_dim' in metadata_df.columns else 384,
                'build_timestamp': metadata_df['timestamp'].iloc[0] if 'timestamp' in metadata_df.columns else 'unknown',
                'categories': metadata_df['category'].unique().tolist() if 'category' in metadata_df.columns else [],
                'languages': metadata_df['language'].unique().tolist() if 'language' in metadata_df.columns else []
            }
        else:
            metadata_info = {
                'total_chunks': 0,
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'embedding_dim': 384,
                'build_timestamp': 'unknown',
                'categories': [],
                'languages': []
            }
        
        # Initialize retriever
        retriever_instance = EmbeddingRetriever()
        logger.info(f"Loaded FAISS index with {metadata_info['total_chunks']} chunks")
        logger.info(f"Using model: {metadata_info['model_name']}")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    
    yield
    
    logger.info("Shutting down BH Assurance RAG API...")

# Create FastAPI app
app = FastAPI(
    title="BH Assurance RAG API",
    description="RAG API for BH Insurance with multilingual support and streaming",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoints
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    )

@app.get("/api/meta", response_model=MetaResponse)
async def get_metadata():
    """Get system metadata and build info"""
    if metadata_info is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return MetaResponse(
        build_info=metadata_info,
        index_stats={
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "status": "ready"
        }
    )

# Search endpoint
@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents without LLM generation"""
    if retriever_instance is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    try:
        # Perform search
        results = retriever_instance.search(
            query=request.query,
            top_k=request.k,
            use_mmr=request.mmr,
            mmr_lambda=request.lambda_
        )
        
        # Convert to Hit objects
        hits = []
        for result in results:
            hits.append(Hit(
                id=result.id,
                score=float(result.score),
                text=result.text,
                filename=result.filename,
                section=result.section,
                category=result.category
            ))
        
        latency = time.time() - start_time
        language = detect_language(request.query)
        
        logger.info(f"[{request_id}] Search completed - Query: '{request.query[:50]}...' | "
                   f"Language: {language} | Results: {len(hits)} | Latency: {latency:.3f}s")
        
        return SearchResponse(
            hits=hits,
            query=request.query,
            used_k=len(hits),
            mmr_enabled=request.mmr
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Streaming chat endpoint
@app.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    """Stream chat response with RAG"""
    if retriever_instance is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Get last user message
    user_messages = [msg for msg in request.messages if msg.role == 'user']
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    last_user_message = user_messages[-1].content
    language = detect_language(last_user_message)
    
    logger.info(f"[{request_id}] Starting RAG chat - Language: {language} | Query: '{last_user_message[:50]}...'")
    
    async def generate_stream():
        try:
            # 1. Retrieve relevant documents
            results = retriever_instance.search(
                query=last_user_message,
                top_k=request.k,
                use_mmr=request.mmr,
                mmr_lambda=request.lambda_
            )
            
            # 2. Build context string
            context_parts = []
            citations = []
            debug_chunks = []
            
            for result in results:
                score = float(result.score)
            filename = result.filename
            section = result.section
            text = result.text
            chunk_id = result.id
            category = result.category
            
            # Add to context (limit length)
            if len('\n'.join(context_parts)) < 8000:  # ~8k tokens limit
                context_parts.append(f"[score={score:.2f}] ({filename} • {section})\n{text}")
            
            # Add citation
            citations.append(Citation(
                id=chunk_id,
                filename=filename,
                section=section,
                score=score
                ))
            
            # Add debug info
            if request.debug:
                debug_chunks.append(Hit(
                    id=chunk_id,
                    score=score,
                    text=text,
                    filename=filename,
                    section=section,
                    category=category
                ))
            
            context = "\n---\n".join(context_parts)
            
            # 3. Initialize LLM client
            llm_client = LLMClient()
            prompt_messages = llm_client.build_prompt(request.messages, context, language)
            
            # 4. Stream LLM response
            async for token in llm_client.stream_chat(prompt_messages, request.temperature):
                yield f"event: token\ndata: {json.dumps({'content': token})}\n\n"
            
            # 5. Send done event with citations
            done_event = DoneEvent(
                citations=citations,
                used_k=len(results),
                provider=LLM_PROVIDER,
                model=LLM_MODEL,
                debug_chunks=debug_chunks if request.debug else None
            )
            
            yield f"event: done\ndata: {done_event.model_dump_json()}\n\n"
            
            latency = time.time() - start_time
            top_files = [c.filename for c in citations[:3]]
            logger.info(f"[{request_id}] Chat completed - Language: {language} | "
                       f"Citations: {len(citations)} | Top files: {top_files} | Latency: {latency:.3f}s")
            
        except Exception as e:
            logger.error(f"[{request_id}] Chat streaming error: {e}")
            error_msg = f"Error: Unable to generate response. {str(e)}"
            yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

# Non-streaming chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""
    if retriever_instance is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Get last user message
    user_messages = [msg for msg in request.messages if msg.role == 'user']
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    last_user_message = user_messages[-1].content
    language = detect_language(last_user_message)
    
    try:
        # Retrieve documents
        results = retriever_instance.search(
            query=last_user_message,
            top_k=request.k,
            use_mmr=request.mmr,
            mmr_lambda=request.lambda_
        )
        
        # Build context and citations
        context_parts = []
        citations = []
        debug_chunks = []
        
        for result in results:
            score = float(result.score)
            filename = result.filename
            section = result.section
            text = result.text
            chunk_id = result.id
            category = result.category
            
            if len('\n'.join(context_parts)) < 8000:
                context_parts.append(f"[score={score:.2f}] ({filename} • {section})\n{text}")
            
            citations.append(Citation(
                id=chunk_id,
                filename=filename,
                section=section,
                score=score
            ))
            
            if request.debug:
                debug_chunks.append(Hit(
                    id=chunk_id,
                    score=score,
                    text=text,
                    filename=filename,
                    section=section,
                    category=category
                ))
        
        context = "\n---\n".join(context_parts)
        
        # Generate response (collect all tokens)
        llm_client = LLMClient()
        prompt_messages = llm_client.build_prompt(request.messages, context, language)
        
        content_parts = []
        async for token in llm_client.stream_chat(prompt_messages, request.temperature):
            content_parts.append(token)
        
        full_content = ''.join(content_parts)
        
        latency = time.time() - start_time
        logger.info(f"[{request_id}] Chat completed - Language: {language} | "
                   f"Citations: {len(citations)} | Latency: {latency:.3f}s")
        
        return ChatResponse(
            content=full_content,
            citations=citations,
            used_k=len(results),
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            debug_chunks=debug_chunks if request.debug else None
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)