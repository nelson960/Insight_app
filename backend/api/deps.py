from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional, Sequence

from backend.core.workspace import get_workspace, Workspace
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
from backend.services.storage import (
    SQLiteConfig,
    QdrantConfig,
    SQLiteMetadataStore,
    QdrantVectorIndex,
    create_sqlite_store,
    create_qdrant_index,
)
from backend.services.connectors import NomicOnnxConfig, NomicOnnxEmbedTextConnector, LlamaSessionManager
from backend.services.search import DocSearchService, FileSearchService

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
    _search_service: Optional[FileSearchService] = None
    _doc_search_service: Optional[DocSearchService] = None
    _nomic_model_dir: Optional[Path] = None

    @classmethod
    def workspace(cls) -> Workspace:
        if cls._workspace is None:
            cls._workspace = get_workspace()
        return cls._workspace

    @classmethod
    def sqlite_store(cls) -> SQLiteMetadataStore:
        if cls._sqlite_store is None:
            with cls._storage_lock:
                if cls._sqlite_store is None:
                    cls._sqlite_store = create_sqlite_store(cls.workspace().db, config=SQLiteConfig())
        return cls._sqlite_store

    @classmethod
    def vector_index(cls) -> QdrantVectorIndex:
        if cls._vector_index is None:
            # Protect local Qdrant initialization from concurrent calls.
            with cls._storage_lock:
                if cls._vector_index is None:
                    cls._vector_index = create_qdrant_index(
                        config=QdrantConfig(
                            collection_name="insight_chunks",
                            path=str(cls.workspace().qdrant),
                        )
                    )
        return cls._vector_index

    @classmethod
    def storage(cls) -> tuple[SQLiteMetadataStore, QdrantVectorIndex]:
        return cls.sqlite_store(), cls.vector_index()

    @classmethod
    def ingestion_pipeline(cls) -> IngestionPipeline:
        if cls._ingestion_pipeline is None:
            sqlite_store = cls.sqlite_store()
            vector_index = cls.vector_index()
            cls._ingestion_pipeline = create_ingestion_pipeline(
                metadata_store=sqlite_store,
                vector_index=vector_index,
                pipeline_config=IngestionPipelineConfig(embedding_model="nomic-embed-text-v1.5", embedding_version=1),
                # Use a writable workspace location (download-on-missing) with a fallback
                # to the bundled repo directory during development.
                nomic_model_dir=cls.nomic_model_dir(),
                use_onnx_embeddings=True,
                nomic_auto_download=True,
            )
        return cls._ingestion_pipeline

    @classmethod
    def ingestion_scheduler(cls) -> IngestionScheduler:
        if cls._ingestion_scheduler is None:
            sqlite_store = cls.sqlite_store()
            cls._ingestion_scheduler = IngestionScheduler(cls.ingestion_pipeline(), sqlite_store)
        return cls._ingestion_scheduler

    @classmethod
    def planner_service(cls) -> PlannerService:
        if cls._planner_service is None:
            sqlite_store = cls.sqlite_store()
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
            sqlite_store = cls.sqlite_store()

            model_path_raw = sqlite_store.get_setting("llm_model_path", "")
            model_path_raw = model_path_raw if isinstance(model_path_raw, str) else ""
            if not model_path_raw.strip():
                raise FileNotFoundError(
                    "No GGUF model configured. Update Settings → Model to select a valid .gguf file."
                )

            model_path = Path(model_path_raw).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. Update Settings → Model to select a valid .gguf file."
                )

            # Context length can be configured via Settings (8k/32k).
            ctx_raw = sqlite_store.get_setting("llm_ctx_size", 32768)
            try:
                ctx_size = int(ctx_raw)
            except Exception:
                ctx_size = 32768
            if ctx_size not in (8192, 32768):
                ctx_size = 32768

            gpu_layers_raw = sqlite_store.get_setting("llm_gpu_layers", 99)
            try:
                gpu_layers = int(gpu_layers_raw)
            except Exception:
                gpu_layers = 99

            persist_dir = Path(get_workspace().base) / "kv_sessions"
            cls._session_manager = LlamaSessionManager(
                str(model_path),
                ctx_size=ctx_size,
                gpu_layers=gpu_layers,
                persist_dir=persist_dir,
                ltm_store=AppDependencies.ltm_store(),
                metadata_store=sqlite_store,
            )
        return cls._session_manager

    @classmethod
    def rag_store(cls) -> RagStore:
        if cls._rag_store is None:
            sqlite_store = cls.sqlite_store()
            vector_index = cls.vector_index()
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
            try:
                client = cls.vector_index().client  # reuse same Qdrant client to avoid lock conflicts
                cls._ltm_store = LtmQdrantStore(client=client, embedder=cls.query_embedder())
            except Exception as exc:
                logger.warning("Falling back to in-memory LTM store: %s", exc)
                cls._ltm_store = LongTermMemoryStore(
                    metadata_store=cls.sqlite_store(),
                    embedder=cls.query_embedder(),
                )
        return cls._ltm_store

    @classmethod
    def key_manager(cls) -> KeyManager:
        if cls._key_manager is None:
            cls._key_manager = KeyManager(cls.workspace())
        return cls._key_manager

    @classmethod
    def search_service(cls) -> FileSearchService:
        if cls._search_service is None:
            sqlite_store = cls.sqlite_store()
            cls._search_service = FileSearchService(sqlite_store)
        return cls._search_service

    @classmethod
    def doc_search_service(cls) -> DocSearchService:
        if cls._doc_search_service is None:
            sqlite_store = cls.sqlite_store()
            cls._doc_search_service = DocSearchService(sqlite_store)
        return cls._doc_search_service

    @classmethod
    def query_embedder(cls) -> Optional[Callable[[str], Sequence[float]]]:
        if cls._query_embedder is not None:
            return cls._query_embedder
        try:
            connector = NomicOnnxEmbedTextConnector(
                model_dir=cls.nomic_model_dir(),
                config=NomicOnnxConfig(),
                auto_download=True,
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

    @classmethod
    def nomic_model_dir(cls) -> Path:
        """
        Resolve the Nomic embedding model directory.

        Preference order:
        1) previously resolved value (cached)
        2) workspace-local (writable) model dir, if it contains required assets
        3) bundled repo model dir under `backend/em_models`, if it contains required assets
        4) workspace-local model dir (will be auto-downloaded on first use)
        """
        if cls._nomic_model_dir is not None:
            return cls._nomic_model_dir

        workspace_dir = Path(cls.workspace().base) / "em_models" / "nomic-embed-text"
        project_root = Path(__file__).resolve().parents[2]
        bundled_dir = project_root / "backend" / "em_models" / "nomic-embed-text"

        def _has_required_assets(base: Path) -> bool:
            return bool((base / "tokenizer.json").exists() and (base / "onnx" / "model.onnx").exists())

        if _has_required_assets(workspace_dir):
            cls._nomic_model_dir = workspace_dir
        elif _has_required_assets(bundled_dir):
            cls._nomic_model_dir = bundled_dir
        else:
            cls._nomic_model_dir = workspace_dir

        return cls._nomic_model_dir

    @classmethod
    def busy_state(cls) -> dict[str, object]:
        """
        Best-effort activity check used to block destructive operations (reset/clean)
        while background work is running (ingestion, streaming, KV snapshot/persist).
        """
        store = cls._sqlite_store
        close_store = False
        if store is None:
            try:
                store = SQLiteMetadataStore(cls.workspace().db, config=SQLiteConfig())
                close_store = True
            except Exception:
                store = None
        active_jobs_db = 0
        if store is not None:
            try:
                active_jobs_db = int(store.count_jobs_with_status(("queued", "running")))
            except Exception:
                active_jobs_db = 0
            if close_store:
                try:
                    store.close()
                except Exception:
                    pass

        ingestion_state: dict[str, int] | None = None
        active_jobs_mem = 0
        if cls._ingestion_scheduler is not None:
            try:
                ingestion_state = cls._ingestion_scheduler.busy_state()
                active_jobs_mem = int(ingestion_state.get("queued", 0)) + int(ingestion_state.get("running", 0))
            except Exception:
                ingestion_state = None
                active_jobs_mem = 0

        active_jobs = max(active_jobs_db, active_jobs_mem)

        llm_state: dict[str, object] | None = None
        llm_busy = False
        if cls._session_manager is not None:
            try:
                llm_state = cls._session_manager.busy_state()
                llm_busy = bool(llm_state.get("busy"))
            except Exception as exc:
                llm_busy = True
                llm_state = {"busy": True, "error": str(exc)}

        busy = bool(active_jobs) or llm_busy
        reasons: list[str] = []
        if active_jobs:
            reasons.append("ingestion")
        if llm_busy:
            reasons.append("llm")

        return {
            "busy": busy,
            "reasons": reasons,
            "active_jobs": active_jobs,
            "ingestion": ingestion_state or {"queued": 0, "running": 0, "timed_out": 0},
            "llm": llm_state or {"busy": False},
        }

    @classmethod
    def reset_all(cls, *, confirm: bool = False, keep_em_models: bool = False) -> None:
        """
        Delete all user data and reset in-memory singletons.

        This closes local Qdrant and SQLite connections so the workspace folder can
        be removed safely, then recreates the workspace structure. The caller
        should restart the engine/app after calling this.
        """
        if not confirm:
            raise ValueError("Reset not confirmed.")

        # Stop model/session workers first (they can hold open files under kv_sessions).
        if cls._session_manager is not None:
            try:
                cls._session_manager._shutdown_snapshot_worker()
            except Exception:
                pass
            try:
                cls._session_manager._shutdown_persist_worker()
            except Exception:
                pass
            cls._session_manager = None

        # Close Qdrant local file lock.
        if cls._vector_index is not None:
            try:
                cls._vector_index.client.close()
            except Exception:
                pass
            cls._vector_index = None

        # Close SQLite connection.
        if cls._sqlite_store is not None:
            try:
                cls._sqlite_store.close()
            except Exception:
                pass
            cls._sqlite_store = None

        # Drop other cached singletons.
        cls._ingestion_pipeline = None
        cls._planner_service = None
        cls._key_manager = None
        cls._ingestion_scheduler = None
        cls._query_embedder = None
        cls._rag_store = None
        cls._ltm_store = None
        cls._search_service = None
        cls._doc_search_service = None

        ws = cls.workspace()
        ws.reset(confirm=True, keep_em_models=keep_em_models)
        cls._workspace = None
