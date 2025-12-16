from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional, Sequence

from backend.core.workspace import get_workspace, Workspace
from backend.services.bootstrap import create_storage_backends
from backend.services.ingestion import (
    IngestionPipeline,
    IngestionPipelineConfig,
    create_ingestion_pipeline,
)
from backend.services.ingestion.scheduler import IngestionScheduler
from backend.services.planner import PlannerService
from backend.services.planner.orchestrator import InsightOrchestrator
from backend.services.memory.ltm_store import LongTermMemoryStore
from backend.services.memory.ltm_qdrant_store import LtmQdrantStore
from backend.services.retrieval.rag_store import RagStore
from backend.services.retrieval import RetrievalService
from backend.services.security import KeyManager
from backend.services.storage import SQLiteConfig, QdrantConfig, SQLiteMetadataStore, QdrantVectorIndex
from backend.services.connectors import NomicOnnxConfig, NomicOnnxEmbedTextConnector, LlamaSessionManager

logger = logging.getLogger(__name__)


class AppDependencies:
    _workspace: Optional[Workspace] = None
    _sqlite_store: Optional[SQLiteMetadataStore] = None
    _vector_index: Optional[QdrantVectorIndex] = None
    _storage_lock = threading.Lock()
    _ingestion_pipeline: Optional[IngestionPipeline] = None
    _planner_service: Optional[PlannerService] = None
    _key_manager: Optional[KeyManager] = None
    _ingestion_scheduler: Optional[IngestionScheduler] = None
    _query_embedder: Optional[Callable[[str], Sequence[float]]] = None
    _session_manager: Optional[LlamaSessionManager] = None
    _rag_store: Optional[RagStore] = None
    _ltm_store: Optional[Any] = None

    @classmethod
    def workspace(cls) -> Workspace:
        if cls._workspace is None:
            cls._workspace = get_workspace()
        return cls._workspace

    @classmethod
    def storage(cls) -> tuple[SQLiteMetadataStore, QdrantVectorIndex]:
        if cls._sqlite_store is None or cls._vector_index is None:
            # Protect local Qdrant initialization from concurrent calls.
            with cls._storage_lock:
                if cls._sqlite_store is None or cls._vector_index is None:
                    cls._sqlite_store, cls._vector_index = create_storage_backends(
                        sqlite_path=cls.workspace().db,
                        sqlite_config=SQLiteConfig(),
                        qdrant_config=QdrantConfig(
                            collection_name="insight_chunks",
                            path=str(cls.workspace().qdrant),
                        ),
                    )
        return cls._sqlite_store, cls._vector_index

    @classmethod
    def ingestion_pipeline(cls) -> IngestionPipeline:
        if cls._ingestion_pipeline is None:
            sqlite_store, vector_index = cls.storage()
            base_dir = Path(__file__).resolve().parents[2]
            cls._ingestion_pipeline = create_ingestion_pipeline(
                metadata_store=sqlite_store,
                vector_index=vector_index,
                pipeline_config=IngestionPipelineConfig(embedding_model="nomic-embed-text-v1.5", embedding_version=1),
                # Use an absolute path so desktop/IPC mode doesn't depend on cwd.
                nomic_model_dir=base_dir / "backend" / "em_models" / "nomic-embed-text",
                use_onnx_embeddings=True,
                nomic_auto_download=False,
            )
        return cls._ingestion_pipeline

    @classmethod
    def ingestion_scheduler(cls) -> IngestionScheduler:
        if cls._ingestion_scheduler is None:
            sqlite_store, _ = cls.storage()
            cls._ingestion_scheduler = IngestionScheduler(cls.ingestion_pipeline(), sqlite_store)
        return cls._ingestion_scheduler

    @classmethod
    def planner_service(cls) -> PlannerService:
        if cls._planner_service is None:
            sqlite_store, _ = cls.storage()
            orchestrator = InsightOrchestrator(
                session_mgr=cls.session_manager(),
                rag_store=cls.rag_store(),
                ltm_store=cls.ltm_store(),
                metadata_store=sqlite_store,
            )
            cls._planner_service = PlannerService(orchestrator=orchestrator)
        return cls._planner_service

    @classmethod
    def session_manager(cls) -> LlamaSessionManager:
        if cls._session_manager is None:
            # Hardcoded local model path; ensure the file exists.
            model_path = Path(__file__).resolve().parents[2] / "models" / "Llama-3.1-8B-Instruct-q4_k_m.gguf"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            persist_dir = Path(get_workspace().base) / "kv_sessions"
            sqlite_store, _ = cls.storage()
            cls._session_manager = LlamaSessionManager(
                str(model_path),
                ctx_size=32768,
                gpu_layers=99,
                persist_dir=persist_dir,
                ltm_store=AppDependencies.ltm_store(),
                metadata_store=sqlite_store,
            )
        return cls._session_manager

    @classmethod
    def rag_store(cls) -> RagStore:
        if cls._rag_store is None:
            sqlite_store, vector_index = cls.storage()
            retrieval = RetrievalService(
                qdrant_client=vector_index.client,
                collection_name=vector_index.collection_name,
                metadata_store=sqlite_store,
            )
            cls._rag_store = RagStore(retrieval_service=retrieval, embedder=cls.query_embedder())
        return cls._rag_store

    @classmethod
    def ltm_store(cls) -> LongTermMemoryStore:
        if cls._ltm_store is None:
            _, vector_index = cls.storage()
            client = vector_index.client  # reuse same Qdrant client to avoid lock conflicts
            cls._ltm_store = LtmQdrantStore(client=client, embedder=cls.query_embedder())
        return cls._ltm_store

    @classmethod
    def key_manager(cls) -> KeyManager:
        if cls._key_manager is None:
            cls._key_manager = KeyManager(cls.workspace())
        return cls._key_manager

    @classmethod
    def query_embedder(cls) -> Optional[Callable[[str], Sequence[float]]]:
        if cls._query_embedder is not None:
            return cls._query_embedder
        try:
            base_dir = Path(__file__).resolve().parents[2]
            connector = NomicOnnxEmbedTextConnector(
                model_dir=base_dir / "backend" / "em_models" / "nomic-embed-text",
                config=NomicOnnxConfig(),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialize local query embedder: %s", exc)
            cls._query_embedder = None
            return cls._query_embedder

        model_name = "nomic-embed-text-v1.5"

        def _embed(text: str) -> Sequence[float]:
            vectors = connector.embed(model_name, [text])
            if not vectors or not vectors[0]:
                raise RuntimeError("Local query embedder returned empty vector.")
            return vectors[0]

        cls._query_embedder = _embed
        return cls._query_embedder
