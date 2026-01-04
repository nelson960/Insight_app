from .chat import router as chat_router
from .docs import router as docs_router
from .files import router as files_router
from .search import router as search_router
from .settings import router as settings_router

__all__ = ["chat_router", "docs_router", "files_router", "search_router", "settings_router"]
