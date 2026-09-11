"""
RAG Chatbot Service — LlamaIndex Hybrid Retrieval + Gemini Generation
Semantic (pgvector HNSW) + Keyword (PostgreSQL FTS) → RRF fusion → Gemini
"""

import os
import logging
import warnings
import psycopg2
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from django.conf import settings

logging.getLogger("llama_index.core.settings").setLevel(logging.ERROR)

from google import genai
from google.genai import types as genai_types

from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from .query_builder import build_rag_query

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / '.env', override=True)

warnings.filterwarnings("ignore")

# LlamaIndex global settings — embedding only, no LLM
LlamaSettings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)
LlamaSettings.llm = None


class HybridRetrieverWithRRF(BaseRetriever):
    """
    Semantic (pgvector HNSW) + Keyword (PostgreSQL FTS) → RRF fusion.

    RRF score = alpha * (1 / (k + rank_semantic))
              + (1-alpha) * (1 / (k + rank_keyword))
    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        db_config: dict,
        semantic_top_k: int = 20,
        keyword_top_k: int = 20,
        final_top_k: int = 10,
        rrf_k: int = 60,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.db_config = db_config
        self.semantic_top_k = semantic_top_k
        self.keyword_top_k = keyword_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.alpha = alpha

    def _keyword_search(self, query_text: str) -> List[dict]:
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, text, metadata_,
                   ts_rank_cd(to_tsvector('simple', text),
                               plainto_tsquery('simple', %s)) AS rank
            FROM data_pdf_chunks
            WHERE to_tsvector('simple', text) @@ plainto_tsquery('simple', %s)
            ORDER BY rank DESC
            LIMIT %s;
            """,
            (query_text, query_text, self.keyword_top_k),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"id": r[0], "text": r[1], "metadata": r[2], "rank": r[3]} for r in rows]

    def _rrf_fusion(
        self,
        semantic_results: List[NodeWithScore],
        keyword_results: List[dict],
    ) -> List[NodeWithScore]:
        scores: dict = {}

        for rank, node in enumerate(semantic_results, start=1):
            nid = node.node.node_id
            scores[nid] = {
                "node": node,
                "score": self.alpha * (1.0 / (self.rrf_k + rank)),
            }

        for rank, result in enumerate(keyword_results, start=1):
            nid = result["id"]
            kw_score = (1 - self.alpha) * (1.0 / (self.rrf_k + rank))
            if nid in scores:
                scores[nid]["score"] += kw_score
            else:
                scores[nid] = {
                    "node": NodeWithScore(
                        node=TextNode(
                            text=result["text"],
                            id_=nid,
                            metadata=result.get("metadata") or {},
                        ),
                        score=kw_score,
                    ),
                    "score": kw_score,
                }

        top = sorted(scores.values(), key=lambda x: x["score"], reverse=True)[: self.final_top_k]
        for item in top:
            item["node"].score = item["score"]
        return [item["node"] for item in top]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_text = query_bundle.query_str
        semantic = self.vector_retriever.retrieve(query_text)
        keyword = self._keyword_search(query_text)
        return self._rrf_fusion(semantic, keyword)


class RAGChatbotService:
    """RAG chatbot: LlamaIndex hybrid retrieval + Google Gemini generation."""

    def __init__(self):
        self.config = settings.RAG_CHATBOT_CONFIG
        self.retriever = None
        self.gemini_client = None
        self.gemini_model_name = None
        self.gemini_temperature = None
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return

        try:
            print("[RAG] Initializing LlamaIndex Hybrid RAG...")

            db_config = {
                "host":     self.config["DB_HOST"],
                "port":     self.config["DB_PORT"],
                "user":     self.config["DB_USER"],
                "password": self.config["DB_PASSWORD"],
                "dbname":   self.config["DB_NAME"],
            }

            print("[RAG] Connecting to PGVectorStore...")
            vector_store = PGVectorStore.from_params(
                host=db_config["host"],
                port=str(db_config["port"]),
                user=db_config["user"],
                password=db_config["password"],
                database=db_config["dbname"],
                table_name=self.config["TABLE_NAME"],
                embed_dim=self.config["EMBED_DIM"],
            )

            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            vector_retriever = index.as_retriever(
                similarity_top_k=self.config["SEMANTIC_TOP_K"]
            )

            self.retriever = HybridRetrieverWithRRF(
                vector_retriever=vector_retriever,
                db_config=db_config,
                semantic_top_k=self.config["SEMANTIC_TOP_K"],
                keyword_top_k=self.config["KEYWORD_TOP_K"],
                final_top_k=self.config["FINAL_TOP_K"],
                rrf_k=self.config["RRF_K"],
                alpha=self.config["ALPHA"],
            )

            print("[RAG] Configuring Gemini...")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            self.gemini_client = genai.Client(api_key=google_api_key)
            self.gemini_model_name = self.config["GEMINI_MODEL"]
            self.gemini_temperature = self.config["GEMINI_TEMPERATURE"]

            self._initialized = True
            print("[RAG] Initialized successfully.")

        except Exception as e:
            print(f"[RAG] ERROR during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _get_chat_history(self, session_id: int) -> List[dict]:
        """Return last 10 messages as Gemini-format history dicts."""
        from chatbot.models import ChatMessage

        history = []
        messages = ChatMessage.objects.filter(
            session_id=session_id
        ).order_by("created_at")[:10]

        for msg in messages:
            role = "user" if msg.sender == "user" else "model"
            history.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=msg.content)],
                )
            )

        return history

    def _build_system_prompt(self, docs_string: str, context: Dict = None) -> str:
        base_prompt = """Sen, göğüs hastalıkları ve radyoloji alanında uzmanlaşmış "Akciğer Röntgeni Analiz Asistanı"sın.
Görevin: Kullanıcının sağladığı NIH Chest X-ray veri seti üzerinde eğitilmiş görüntü işleme modelinin çıktılarını yorumlamak ve tıbbi bağlamda açıklayıcı bilgiler sunmaktır.

# GÖRÜNTÜ İŞLEME MODELİ SONUÇLARI
"""
        if context and context.get("diagnoses"):
            base_prompt += "\n**Model Tahminleri:**\n"
            for diag in context["diagnoses"][:5]:
                base_prompt += (
                    f"- {diag.get('disease_name', 'N/A')}: "
                    f"%{diag.get('percentage', 0):.1f} "
                    f"({diag.get('risk_level', 'Unknown')})\n"
                )

        if context and context.get("patient"):
            patient = context["patient"]
            base_prompt += "\n**Hasta Bilgileri:**\n"
            base_prompt += f"- Yaş: {patient.get('age', 'N/A')}\n"
            base_prompt += f"- Cinsiyet: {patient.get('gender', 'N/A')}\n"
            base_prompt += f"- Pozisyon: {patient.get('position', 'N/A')}\n"

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

    def build_automatic_query(self, context: Dict = None) -> Dict:
        context = context or {}
        return build_rag_query(
            diagnoses=context.get("diagnoses", []),
            patient=context.get("patient", {}),
            thresholds=getattr(settings, "RAG_CLASS_THRESHOLDS", None),
        )

    def get_response(
        self,
        question: str,
        context: Dict = None,
        session_id: int = None,
        query_metadata: Dict = None,
    ) -> Dict:
        try:
            print(f"[RAG] get_response: {question[:60]}...")

            if not self._initialized:
                self._initialize()

            retrieval_query = question
            if query_metadata:
                retrieval_query = query_metadata["query"]

            print("[RAG] Retrieving documents (hybrid)...")
            query_bundle = QueryBundle(query_str=retrieval_query)
            nodes = self.retriever._retrieve(query_bundle)
            print(f"[RAG] Retrieved {len(nodes)} nodes")
            docs_string = "\n\n".join(n.node.text for n in nodes)

            system_prompt = self._build_system_prompt(docs_string, context)

            history = []
            if session_id:
                history = self._get_chat_history(session_id)
                print(f"[RAG] Chat history: {len(history)} messages")

            # system_prompt is passed via system_instruction in GenerateContentConfig
            gemini_history = list(history)

            print("[RAG] Calling Gemini...")
            gemini_history.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=question)],
                )
            )
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=gemini_history,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.gemini_temperature,
                ),
            )
            print(f"[RAG] Gemini response: {response.text[:100]}...")

            return {
                "success": True,
                "content": response.text,
                "source": "RAG System",
                "confidence": None,
                "rag_query": retrieval_query,
                "rag_query_source": (
                    query_metadata["query_source"] if query_metadata else "user_message"
                ),
                "rag_query_metadata": query_metadata or {
                    "query": retrieval_query,
                    "query_source": "user_message",
                    "query_inputs": {"message": question},
                    "selected_labels": [],
                    "probabilities": {},
                    "thresholds": {},
                    "query_template_version": None,
                },
            }

        except Exception as e:
            print(f"[RAG] ERROR in get_response: {e}")
            import traceback
            traceback.print_exc()
            fallback_response = self._get_fallback_response(question, context, error=e)
            if query_metadata:
                fallback_response.update({
                    "rag_query": query_metadata["query"],
                    "rag_query_source": query_metadata["query_source"],
                    "rag_query_metadata": query_metadata,
                })
            return fallback_response

    def _get_fallback_response(self, question: str, context: Dict = None, error: Exception = None) -> Dict:
        question_lower = question.lower()

        if context and context.get("diagnoses"):
            top = context["diagnoses"][0]
            disease = top.get("disease_name", "Unknown")
            confidence = top.get("percentage", 0)

            if any(kw in question_lower for kw in ["tedavi", "treatment", "protokol"]):
                return {
                    "success": True,
                    "content": f"""**{disease} Tedavi Protokolü**

Model %{confidence:.1f} güvenle {disease} bulgusunu tespit etti.

**Genel Öneriler:**
1. Detaylı radyolojik değerlendirme
2. Klinik korelasyon
3. İlgili uzman konsültasyonu
4. Takip görüntülemesi (gerekirse)

⚠️ **Önemli:** Bu öneriler AI modeli tarafından üretilmiştir. Kesin tanı ve tedavi için mutlaka bir sağlık uzmanına danışın.

*Not: RAG sistemi şu anda yüklenemedi. Genel bilgiler gösteriliyor.*""",
                    "source": "Fallback System",
                    "confidence": None,
                }

        error_str = str(error) if error else ""

        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "credits" in error_str.lower():
            detail = (
                "**Google Gemini API krediniz tükenmiş.**\n\n"
                "Çözüm için:\n"
                "1. https://aistudio.google.com adresine gidin\n"
                "2. Projenizin faturalandırma ayarlarını kontrol edin\n"
                "3. Kredi yükleyin veya ücretsiz kotanızı kontrol edin"
            )
        elif "INVALID_ARGUMENT" in error_str or "API_KEY" in error_str:
            detail = "Google Gemini veya OpenAI API anahtarı geçersiz. `.env` dosyasını kontrol edin."
        elif "connection" in error_str.lower() or "psycopg2" in error_str.lower():
            detail = "PostgreSQL bağlantı hatası. Veritabanının çalıştığından emin olun (port 5410)."
        else:
            detail = f"Hata: {error_str[:200]}" if error_str else "Bilinmeyen hata."

        return {
            "success": False,
            "content": f"Yapay zeka yanıt üretemedi.\n\n{detail}",
            "source": "System Error",
            "confidence": None,
        }


# Singleton
_chatbot_service = None


def get_chatbot_service() -> RAGChatbotService:
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = RAGChatbotService()
    return _chatbot_service
