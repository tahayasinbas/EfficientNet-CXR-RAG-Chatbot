"""
RAG Chatbot Service with Memory Support
Direct integration with local RAG system using PostgreSQL + pgvector
"""

import os
import torch
import warnings
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
from django.conf import settings
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Get the base directory (kds_django_fantezi/)
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'

# Load environment variables from .env file
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Ensure GOOGLE_API_KEY is loaded
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key
    print("[RAG] GOOGLE_API_KEY loaded from .env")
else:
    print("[RAG] ERROR: GOOGLE_API_KEY not found in .env file!")

# Suppress warnings
warnings.filterwarnings("ignore")


class HybridRetriever:
    """Hybrid retriever combining BM25 (keyword) and semantic search"""

    def __init__(self, bm25_retriever, vector_retriever, bm25_weight=0.4):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.bm25_weight = bm25_weight

    def invoke(self, query: str, k: int = 5):
        """
        Retrieve documents using hybrid search

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of relevant documents
        """
        # Get results from both retrievers
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        # Calculate scores using Reciprocal Rank Fusion
        doc_scores = {}

        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            score = self.bm25_weight * (1 / (rank + 1))
            doc_scores[content] = doc_scores.get(content, 0) + score

        for rank, doc in enumerate(vector_docs):
            content = doc.page_content
            score = (1 - self.bm25_weight) * (1 / (rank + 1))
            if content in doc_scores:
                doc_scores[content] += score
            else:
                doc_scores[content] = score

        # Sort by score
        sorted_contents = sorted(
            doc_scores.keys(),
            key=lambda x: doc_scores[x],
            reverse=True
        )

        # Return documents
        all_docs_dict = {doc.page_content: doc for doc in bm25_docs + vector_docs}
        return [all_docs_dict[content] for content in sorted_contents[:k]]

    def get_relevant_documents(self, query: str):
        """Alias for invoke (for compatibility)"""
        return self.invoke(query)


class RAGChatbotService:
    """Service for RAG-based chatbot using local PostgreSQL and Gemini with memory support"""

    def __init__(self):
        self.config = settings.RAG_CHATBOT_CONFIG
        self.retriever = None
        self.llm = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of heavy components"""
        if self._initialized:
            return

        try:
            print(f"[RAG] Initializing RAG Chatbot on {self.device.upper()}...")

            # 1. Initialize embedding model
            print(f"[RAG] Loading embedding model: {self.config['MODEL_NAME']}...")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config['MODEL_NAME'],
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )

            # 2. Connect to PostgreSQL vector store
            print("[RAG] Connecting to PostgreSQL vector store...")
            vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=self.config['COLLECTION_NAME'],
                connection=self.config['CONNECTION_STRING'],
                use_jsonb=True,
            )

            # 3. Load all documents for BM25
            print("[RAG] Loading documents for BM25 retriever...")
            all_docs = vectorstore.similarity_search("", k=50000)  # Load all docs
            print(f"[RAG] Loaded {len(all_docs)} documents")

            # 4. Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 5

            # 5. Create vector retriever
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            # 6. Create hybrid retriever
            self.retriever = HybridRetriever(
                bm25_retriever=bm25_retriever,
                vector_retriever=vector_retriever,
                bm25_weight=0.4
            )

            # 7. Initialize Gemini LLM (exactly like in hafizaliRag.ipynb)
            print("[RAG] Initializing Google Gemini...")

            self.llm = ChatGoogleGenerativeAI(
                model=self.config['GEMINI_MODEL'],
                temperature=self.config['GEMINI_TEMPERATURE'],
                convert_system_message_to_human=True
            )

            self._initialized = True
            print("[RAG] RAG Chatbot initialized successfully!")

        except Exception as e:
            print(f"[RAG] ERROR initializing RAG Chatbot: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _get_chat_history(self, session_id: int) -> List:
        """
        Get chat history from database and convert to LangChain message format

        Args:
            session_id: Chat session ID

        Returns:
            List of LangChain messages
        """
        from chatbot.models import ChatMessage

        messages = []

        # Get last 10 messages for context (to avoid token limits)
        chat_messages = ChatMessage.objects.filter(
            session_id=session_id
        ).order_by('created_at')[:10]

        for msg in chat_messages:
            if msg.sender == 'user':
                messages.append(HumanMessage(content=msg.content))
            elif msg.sender == 'ai':
                messages.append(AIMessage(content=msg.content))

        return messages

    def _build_system_prompt(self, docs_string: str, context: Dict = None) -> str:
        """
        Build system prompt with medical documents and X-ray analysis context

        Args:
            docs_string: Retrieved medical documents
            context: Patient info and diagnosis results

        Returns:
            System prompt string
        """
        # Base system prompt for X-ray analysis
        base_prompt = """Sen, göğüs hastalıkları ve radyoloji alanında uzmanlaşmış "Akciğer Röntgeni Analiz Asistanı"sın.
Görevin: Kullanıcının sağladığı NIH Chest X-ray veri seti üzerinde eğitilmiş görüntü işleme modelinin çıktılarını yorumlamak ve tıbbi bağlamda açıklayıcı bilgiler sunmaktır.

# GÖRÜNTÜ İŞLEME MODELİ SONUÇLARI
"""

        # Add X-ray diagnosis results if available
        if context and 'diagnoses' in context and context['diagnoses']:
            base_prompt += "\n**Model Tahminleri:**\n"
            for diag in context['diagnoses'][:5]:  # Top 5 diagnoses
                disease_name = diag.get('disease_name', 'N/A')
                percentage = diag.get('percentage', 0)
                risk_level = diag.get('risk_level', 'Unknown')
                base_prompt += f"- {disease_name}: %{percentage:.1f} ({risk_level})\n"

        # Add patient information
        if context and 'patient' in context:
            patient = context['patient']
            base_prompt += f"\n**Hasta Bilgileri:**\n"
            base_prompt += f"- Yaş: {patient.get('age', 'N/A')}\n"
            base_prompt += f"- Cinsiyet: {patient.get('gender', 'N/A')}\n"
            base_prompt += f"- Pozisyon: {patient.get('position', 'N/A')}\n"

        # Add rules and guidelines
        base_prompt += """
# KURALLAR VE DAVRANIŞLAR
1. **Analiz Odaklı Ol:** Görüntü işleme modelinden gelen yüksek olasılıklı (%50 üzeri) hastalıkları birincil bulgu olarak ele al.
2. **NIH Veri Seti Bilgisi:** Aşağıdaki patolojiler hakkında detaylı bilgiye sahipsin: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia.
3. **Tıbbi Açıklama:** Tespit edilen hastalığın ne olduğunu, röntgende genelde nasıl göründüğünü ve potansiyel klinik önemini açıkla.
4. **Tedavi Önerisi (Dikkatli Ol):** Kesin reçete yazma. Genel tıbbi protokollerden bahset (Örn: "Genellikle antibiyotik tedavisi uygulanır ancak hekim kararı esastır").
5. **Ton:** Profesyonel, sakin, empatik ve kanıta dayalı.
6. **Hafıza:** Önceki konuşmaları hatırla ve bağlam içinde yanıt ver.

# GÜVENLİK VE YASAL UYARI (ÇOK ÖNEMLİ)
- Asla "Sende kanser var" veya "Kesinlikle hastasın" gibi ifadeler kullanma.
- Her zaman cümlelerini "Modelin analizine göre...", "Bulgular ... yönünde işaret veriyor" şeklinde kur.
- Cevabının sonuna mutlaka şunu ekle: "Ben bir yapay zeka asistanıyım. Bu sonuçlar bir ön tarama niteliğindedir ve kesin tıbbi teşhis yerine geçmez. Lütfen sonuçları uzman bir radyolog veya göğüs hastalıkları uzmanı ile değerlendirin."

# CEVAP FORMATI
- **Özet Bulgular:** Modelin en yüksek tahminlerini listele.
- **Detaylı Analiz:** Tespit edilen durumların tıbbi açıklaması.
- **Olası Sonraki Adımlar:** (Örn: BT taraması, kan testi vb. önerileri).
- **Yasal Uyarı:** Standart uyarı metni.

# İLGİLİ TIBBİ DOKÜMANLAR
"""
        base_prompt += docs_string

        return base_prompt

    def get_response(self, question: str, context: Dict = None, session_id: int = None) -> Dict:
        """
        Get AI response using RAG system with memory support

        Args:
            question: User's question
            context: Optional context (diagnosis results, patient info)
            session_id: Chat session ID for memory

        Returns:
            dict: AI response with content and metadata
        """
        try:
            print(f"[RAG] get_response called with question: {question[:50]}...")

            # Initialize if not already done
            if not self._initialized:
                print("[RAG] Initializing chatbot...")
                self._initialize()

            # Retrieve relevant documents
            print("[RAG] Retrieving relevant documents...")
            docs = self.retriever.invoke(question)
            print(f"[RAG] Retrieved {len(docs)} documents")
            docs_string = "\n\n".join([doc.page_content for doc in docs])

            # Build system prompt
            print("[RAG] Building system prompt...")
            system_prompt = self._build_system_prompt(docs_string, context)

            # Get chat history from database
            chat_history = []
            if session_id:
                print(f"[RAG] Loading chat history for session {session_id}...")
                chat_history = self._get_chat_history(session_id)
                print(f"[RAG] Loaded {len(chat_history)} previous messages")

            # Build message chain: [System Prompt] + [Chat History] + [New Question]
            messages = [
                SystemMessage(content=system_prompt)
            ] + chat_history + [
                HumanMessage(content=question)
            ]

            # Get response from Gemini
            print("[RAG] Calling Gemini API...")
            ai_msg = self.llm.invoke(messages)
            print(f"[RAG] Gemini response received: {ai_msg.content[:100]}...")

            return {
                'success': True,
                'content': ai_msg.content,
                'source': 'RAG System',
                'confidence': None
            }

        except Exception as e:
            print(f"[RAG] ERROR in get_response: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_response(question, context)

    def _get_fallback_response(self, question: str, context: Dict = None) -> Dict:
        """
        Generate fallback response when RAG system fails
        """
        question_lower = question.lower()

        # Check for diagnosis context
        if context and 'diagnoses' in context and context['diagnoses']:
            top_diagnosis = context['diagnoses'][0]
            disease = top_diagnosis.get('disease_name', 'Unknown')
            confidence = top_diagnosis.get('percentage', 0)

            if any(keyword in question_lower for keyword in ['tedavi', 'treatment', 'protokol']):
                return {
                    'success': True,
                    'content': f"""**{disease} Tedavi Protokolü**

Model %{confidence:.1f} güvenle {disease} bulgusunu tespit etti.

**Genel Öneriler:**
1. Detaylı radyolojik değerlendirme
2. Klinik korelasyon
3. İlgili uzman konsültasyonu
4. Takip görüntülemesi (gerekirse)

⚠️ **Önemli:** Bu öneriler AI modeli tarafından üretilmiştir. Kesin tanı ve tedavi için mutlaka bir sağlık uzmanına danışın.

*Not: RAG sistemi şu anda yüklenemedi. Genel bilgiler gösteriliyor.*""",
                    'source': 'Fallback System',
                    'confidence': None
                }

        # Generic fallback
        return {
            'success': False,
            'content': """RAG sistemi şu anda başlatılamadı.

Olası nedenler:
- PostgreSQL veritabanı bağlantı hatası
- Embedding modeli yüklenemedi
- Google Gemini API anahtarı geçersiz

Lütfen sistem yöneticisiyle iletişime geçin.""",
            'source': 'System Error',
            'confidence': None
        }


# Singleton instance
_chatbot_service = None


def get_chatbot_service():
    """Get or create chatbot service singleton"""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = RAGChatbotService()
    return _chatbot_service
