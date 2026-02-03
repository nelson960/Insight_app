from __future__ import annotations

import logging
import sys
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

from backend.core.workspace import get_workspace, Workspace

# Boot trace (optional import)
try:
    from backend.services.boot_trace import log_boot_step, log_boot_error
    _boot_trace_available = True
except Exception:
    _boot_trace_available = False
if TYPE_CHECKING:
    from backend.services.ingestion import IngestionPipeline
    from backend.services.ingestion.scheduler import IngestionScheduler
    from backend.services.planner.service import PlannerService
    from backend.services.memory.ltm_store import LongTermMemoryStore
    from backend.services.retrieval.rag_store import RagStore
    from backend.services.security.key_manager import KeyManager
    from backend.services.storage import SQLiteMetadataStore, QdrantVectorIndex
    from backend.services.connectors.llama_session_manager import LlamaSessionManager
    from backend.services.search.service import DocSearchService, FileSearchService

logger = logging.getLogger(__name__)


class _AppContainer:
    def __init__(self) -> None:
        self._workspace: Optional[Workspace] = None
        self._sqlite_store: Optional[SQLiteMetadataStore] = None
        self._vector_index: Optional[QdrantVectorIndex] = None
        self._storage_lock = threading.Lock()
        self._ingestion_pipeline: Optional[IngestionPipeline] = None
        self._planner_service: Optional[PlannerService] = None
        self._key_manager: Optional[KeyManager] = None
        self._ingestion_scheduler: Optional[IngestionScheduler] = None
        self._query_embedder: Optional[Callable[[str], Sequence[float]]] = None
        self._session_manager: Optional[LlamaSessionManager] = None
        self._rag_store: Optional[RagStore] = None
        self._ltm_store: Optional[Any] = None
        self._search_service: Optional[FileSearchService] = None
        self._doc_search_service: Optional[DocSearchService] = None
        self._nomic_model_dir: Optional[Path] = None


_current_container: ContextVar[Optional[_AppContainer]] = ContextVar("insight_deps", default=None)
_default_container: Optional[_AppContainer] = None


def _get_container() -> _AppContainer:
    container = _current_container.get()
    if container is not None:
        return container
    global _default_container
    if _default_container is None:
        _default_container = _AppContainer()
    return _default_container


def bind_container(container: Optional[_AppContainer]) -> None:
    global _default_container
    _default_container = container


def create_container() -> _AppContainer:
    return _AppContainer()


def set_request_container(container: Optional[_AppContainer]):
    return _current_container.set(container)


def reset_request_container(token) -> None:
    _current_container.reset(token)


def peek_sqlite_store() -> Optional["SQLiteMetadataStore"]:
    return _get_container()._sqlite_store


def peek_session_manager() -> Optional["LlamaSessionManager"]:
    return _get_container()._session_manager


class AppDependencies:

    @classmethod
    def workspace(cls) -> Workspace:
        container = _get_container()
        if container._workspace is None:
            container._workspace = get_workspace()
        return container._workspace

    @classmethod
    def sqlite_store(cls) -> SQLiteMetadataStore:
        container = _get_container()
        if container._sqlite_store is None:
            with container._storage_lock:
                if container._sqlite_store is None:
                    from backend.services.storage.sqlite_store import SQLiteConfig, create_sqlite_store

                    container._sqlite_store = create_sqlite_store(cls.workspace().db, config=SQLiteConfig())
        return container._sqlite_store

    @classmethod
    def vector_index(cls) -> QdrantVectorIndex:
        container = _get_container()
        if container._vector_index is None:
            # Protect local Qdrant initialization from concurrent calls.
            with container._storage_lock:
                if container._vector_index is None:
                    from backend.services.storage import QdrantConfig, create_qdrant_index

                    container._vector_index = create_qdrant_index(
                        config=QdrantConfig(
                            collection_name="insight_chunks",
                            path=str(cls.workspace().qdrant),
                        )
                    )
        return container._vector_index

    @classmethod
    def storage(cls) -> tuple[SQLiteMetadataStore, QdrantVectorIndex]:
        return cls.sqlite_store(), cls.vector_index()

    @classmethod
    def ingestion_pipeline(cls) -> IngestionPipeline:
        container = _get_container()
        if container._ingestion_pipeline is None:
            from backend.services.ingestion import (
                IngestionPipelineConfig,
                create_ingestion_pipeline,
            )

            sqlite_store = cls.sqlite_store()
            vector_index = cls.vector_index()
            container._ingestion_pipeline = create_ingestion_pipeline(
                metadata_store=sqlite_store,
                vector_index=vector_index,
                pipeline_config=IngestionPipelineConfig(embedding_model="nomic-embed-text-v1.5", embedding_version=1),
                # Use a writable workspace location (download-on-missing) with a fallback
                # to the bundled repo directory during development.
                nomic_model_dir=cls.nomic_model_dir(),
                use_onnx_embeddings=True,
                nomic_auto_download=False,
            )
        return container._ingestion_pipeline

    @classmethod
    def ingestion_scheduler(cls) -> IngestionScheduler:
        container = _get_container()
        if container._ingestion_scheduler is None:
            from backend.services.ingestion.scheduler import IngestionScheduler

            sqlite_store = cls.sqlite_store()
            container._ingestion_scheduler = IngestionScheduler(cls.ingestion_pipeline(), sqlite_store)
        return container._ingestion_scheduler

    @classmethod
    def planner_service(cls) -> PlannerService:
        container = _get_container()
        if container._planner_service is None:
            from backend.services.planner.orchestrator import InsightOrchestrator
            from backend.services.planner.service import PlannerService

            sqlite_store = cls.sqlite_store()
            orchestrator = InsightOrchestrator(
                session_mgr=cls.session_manager(),
                rag_store=cls.rag_store(),
                ltm_store=cls.ltm_store(),
                metadata_store=sqlite_store,
            )
            container._planner_service = PlannerService(orchestrator=orchestrator)
        return container._planner_service

    @classmethod
    def session_manager(cls) -> LlamaSessionManager:
        container = _get_container()
        if container._session_manager is None:
            from backend.services.connectors.llama_session_manager import LlamaSessionManager

            if _boot_trace_available:
                try:
                    log_boot_step("session_manager_init_start")
                except Exception:
                    pass

            sqlite_store = cls.sqlite_store()

            model_path_raw = sqlite_store.get_setting("llm_model_path", "")
            model_path_raw = model_path_raw if isinstance(model_path_raw, str) else ""
            if not model_path_raw.strip():
                error = "No GGUF model configured. Update Settings → Model to select a valid .gguf file."
                if _boot_trace_available:
                    try:
                        log_boot_step("session_manager_error", error="no_model_configured")
                    except Exception:
                        pass
                raise FileNotFoundError(error)

            model_path = Path(model_path_raw).expanduser()
            if not model_path.exists():
                error = f"Model not found at {model_path}. Update Settings → Model to select a valid .gguf file."
                if _boot_trace_available:
                    try:
                        log_boot_step("session_manager_error", error="model_not_found", model_path=str(model_path))
                    except Exception:
                        pass
                raise FileNotFoundError(error)

            # Require a validated model record (no silent fallback).
            try:
                from backend.services.llama_templates import model_record_is_current

                record = sqlite_store.get_setting("llm_model_record_json")
                if not model_record_is_current(record, model_path):
                    error = (
                        "Model validation record missing or outdated. "
                        "Open Settings → Model and apply to validate this GGUF."
                    )
                    if _boot_trace_available:
                        try:
                            log_boot_step(
                                "session_manager_error",
                                error="model_record_missing_or_stale",
                                model_path=str(model_path),
                            )
                        except Exception:
                            pass
                    raise FileNotFoundError(error)
            except FileNotFoundError:
                raise
            except Exception as exc:
                logger.warning("Model record check failed: %s", exc)

            if _boot_trace_available:
                try:
                    log_boot_step("session_manager_loading", model_path=str(model_path))
                except Exception:
                    pass

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

            persist_dir = Path(cls.workspace().base) / "kv_sessions"

            try:
                container._session_manager = LlamaSessionManager(
                    str(model_path),
                    ctx_size=ctx_size,
                    gpu_layers=gpu_layers,
                    persist_dir=persist_dir,
                    ltm_store=AppDependencies.ltm_store(),
                    metadata_store=sqlite_store,
                )
                if _boot_trace_available:
                    try:
                        log_boot_step("session_manager_loaded", status="OK")
                    except Exception:
                        pass
            except Exception as e:
                if _boot_trace_available:
                    try:
                        log_boot_error("session_manager_init", e)
                    except Exception:
                        pass
                raise
        return container._session_manager

    @classmethod
    def rag_store(cls) -> RagStore:
        container = _get_container()
        if container._rag_store is None:
            from backend.services.retrieval import RetrievalService
            from backend.services.retrieval.rag_store import RagStore

            sqlite_store = cls.sqlite_store()
            vector_index = cls.vector_index()
            retrieval = RetrievalService(
                qdrant_client=vector_index.client,
                collection_name=vector_index.collection_name,
                metadata_store=sqlite_store,
            )
            container._rag_store = RagStore(retrieval_service=retrieval, embedder=cls.query_embedder())
        return container._rag_store

    @classmethod
    def ltm_store(cls) -> LongTermMemoryStore:
        container = _get_container()
        if container._ltm_store is None:
            from backend.services.memory.ltm_qdrant_store import LtmQdrantStore
            from backend.services.memory.ltm_sqlite_store import LtmSqliteStore
            from backend.services.memory.ltm_store import LongTermMemoryStore

            try:
                client = cls.vector_index().client  # reuse same Qdrant client to avoid lock conflicts
                container._ltm_store = LtmQdrantStore(client=client, embedder=cls.query_embedder())
            except Exception as exc:
                logger.warning("Falling back to SQLite LTM store: %s", exc)
                try:
                    container._ltm_store = LtmSqliteStore(
                        metadata_store=cls.sqlite_store(),
                        embedder=cls.query_embedder(),
                    )
                except Exception as exc2:
                    logger.warning("Falling back to in-memory LTM store: %s", exc2)
                    container._ltm_store = LongTermMemoryStore(
                        metadata_store=cls.sqlite_store(),
                        embedder=cls.query_embedder(),
                    )
        return container._ltm_store

    @classmethod
    def key_manager(cls) -> KeyManager:
        container = _get_container()
        if container._key_manager is None:
            from backend.services.security.key_manager import KeyManager

            container._key_manager = KeyManager(cls.workspace())
        return container._key_manager

    @classmethod
    def search_service(cls) -> FileSearchService:
        container = _get_container()
        if container._search_service is None:
            from backend.services.search.service import FileSearchService

            sqlite_store = cls.sqlite_store()
            container._search_service = FileSearchService(sqlite_store)
        return container._search_service

    @classmethod
    def doc_search_service(cls) -> DocSearchService:
        container = _get_container()
        if container._doc_search_service is None:
            from backend.services.search.doc_search import DocSearchService

            sqlite_store = cls.sqlite_store()
            container._doc_search_service = DocSearchService(sqlite_store)
        return container._doc_search_service

    @classmethod
    def query_embedder(cls) -> Optional[Callable[[str], Sequence[float]]]:
        container = _get_container()
        if container._query_embedder is not None:
            return container._query_embedder

        if _boot_trace_available:
            try:
                log_boot_step("query_embedder_init_start")
            except Exception:
                pass

        try:
            from backend.services.connectors.nomic_onnx import (
                NomicOnnxConfig,
                NomicOnnxEmbedTextConnector,
            )

            if _boot_trace_available:
                try:
                    model_dir = cls.nomic_model_dir()
                    log_boot_step("onnx_connector_init", model_dir=str(model_dir))
                except Exception:
                    model_dir = cls.nomic_model_dir()

            connector = NomicOnnxEmbedTextConnector(
                model_dir=model_dir,
                config=NomicOnnxConfig(),
                # Keep downloads manual; embeddings must be installed via Settings.
                auto_download=False,
            )

            if _boot_trace_available:
                try:
                    log_boot_step("onnx_connector_loaded", status="OK")
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialize local query embedder: %s", exc)
            if _boot_trace_available:
                try:
                    log_boot_error("onnx_connector_init", exc)
                except Exception:
                    pass
            container._query_embedder = None
            return container._query_embedder

        model_name = "nomic-embed-text-v1.5"

        def _embed(text: str) -> Sequence[float]:
            vectors = connector.embed(model_name, [text])
            if not vectors or not vectors[0]:
                raise RuntimeError("Local query embedder returned empty vector.")
            return vectors[0]

        container._query_embedder = _embed
        return container._query_embedder

    @classmethod
    def nomic_model_dir(cls) -> Path:
        """
        Resolve the Nomic embedding model directory.

        Preference order:
        1) previously resolved value (cached)
        2) workspace-local (writable) model dir, if it contains required assets
        3) bundled repo model dir under `backend/em_models`, if it contains required assets (NEVER _MEIPASS in packaged mode)
        4) workspace-local model dir (will be auto-downloaded on first use)
        """
        container = _get_container()
        if container._nomic_model_dir is not None:
            return container._nomic_model_dir

        workspace_dir = Path(cls.workspace().base) / "em_models" / "nomic-embed-text"
        project_root = Path(__file__).resolve().parents[2]
        bundled_dir = project_root / "backend" / "em_models" / "nomic-embed-text"

        # Guardrail: In packaged mode, NEVER use bundled_dir if it's under _MEIPASS
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass and str(bundled_dir).startswith(meipass):
            # Ignore bundled models in _MEIPASS - they're ephemeral
            bundled_dir = None

        def _has_required_assets(base: Path) -> bool:
            if base is None:
                return False
            return bool((base / "tokenizer.json").exists() and (base / "onnx" / "model.onnx").exists())

        if _boot_trace_available:
            try:
                log_boot_step("nomic_model_dir_resolve",
                              workspace_dir=str(workspace_dir),
                              bundled_dir=str(bundled_dir) if bundled_dir else "None",
                              workspace_exists=str(_has_required_assets(workspace_dir)),
                              bundled_exists=str(_has_required_assets(bundled_dir) if bundled_dir else False))
            except Exception:
                pass

        if _has_required_assets(workspace_dir):
            container._nomic_model_dir = workspace_dir
            if _boot_trace_available:
                try:
                    log_boot_step("nomic_model_dir_selected", source="workspace", path=str(workspace_dir))
                except Exception:
                    pass
        elif bundled_dir and _has_required_assets(bundled_dir):
            container._nomic_model_dir = bundled_dir
            if _boot_trace_available:
                try:
                    log_boot_step("nomic_model_dir_selected", source="bundled", path=str(bundled_dir))
                except Exception:
                    pass
        else:
            container._nomic_model_dir = workspace_dir
            if _boot_trace_available:
                try:
                    log_boot_step("nomic_model_dir_selected", source="workspace_will_download", path=str(workspace_dir))
                except Exception:
                    pass

        return container._nomic_model_dir

    @classmethod
    def busy_state(cls) -> dict[str, object]:
        """
        Best-effort activity check used to block destructive operations (reset/clean)
        while background work is running (ingestion, streaming, KV snapshot/persist).
        """
        from backend.services.storage.sqlite_store import SQLiteConfig, SQLiteMetadataStore

        container = _get_container()
        store = container._sqlite_store
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
        if container._ingestion_scheduler is not None:
            try:
                ingestion_state = container._ingestion_scheduler.busy_state()
                active_jobs_mem = int(ingestion_state.get("queued", 0)) + int(ingestion_state.get("running", 0))
            except Exception:
                ingestion_state = None
                active_jobs_mem = 0

        active_jobs = max(active_jobs_db, active_jobs_mem)

        llm_state: dict[str, object] | None = None
        llm_busy = False
        if container._session_manager is not None:
            try:
                llm_state = container._session_manager.busy_state()
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

        container = _get_container()
        # Stop model/session workers first (they can hold open files under kv_sessions).
        if container._session_manager is not None:
            try:
                container._session_manager._shutdown_snapshot_worker()
            except Exception:
                pass
            try:
                container._session_manager._shutdown_persist_worker()
            except Exception:
                pass
            container._session_manager = None

        # Close Qdrant local file lock.
        if container._vector_index is not None:
            try:
                container._vector_index.client.close()
            except Exception:
                pass
            container._vector_index = None

        # Close SQLite connection.
        if container._sqlite_store is not None:
            try:
                container._sqlite_store.close()
            except Exception:
                pass
            container._sqlite_store = None

        # Drop other cached singletons.
        container._ingestion_pipeline = None
        container._planner_service = None
        container._key_manager = None
        container._ingestion_scheduler = None
        container._query_embedder = None
        container._rag_store = None
        container._ltm_store = None
        container._search_service = None
        container._doc_search_service = None

        ws = cls.workspace()
        ws.reset(confirm=True, keep_em_models=keep_em_models)
        container._workspace = None

    @classmethod
    def close_all(cls) -> None:
        """
        Best-effort shutdown of background workers and storage connections.

        This does NOT delete user data; it only releases resources.
        """
        container = _get_container()
        if container._ingestion_scheduler is not None:
            try:
                container._ingestion_scheduler.shutdown()
            except Exception:
                pass
        if container._session_manager is not None:
            try:
                container._session_manager._shutdown_snapshot_worker()
            except Exception:
                pass
            try:
                container._session_manager._shutdown_persist_worker()
            except Exception:
                pass
        if container._vector_index is not None:
            try:
                container._vector_index.client.close()
            except Exception:
                pass
        if container._sqlite_store is not None:
            try:
                container._sqlite_store.close()
            except Exception:
                pass
