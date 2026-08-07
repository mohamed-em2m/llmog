"""
`servers/__init__.py` — exports managers so the rest of the codebase
only ever imports from `servers`.
"""

# Import by name so clients can do:
#   from .servers import LlamaServerManager, VllmServerManager
from .llama_server_manager import LlamaServerManager
from .vllm_server_manager import VllmServerManager

servers_factory = {
    "llama_cpp": LlamaServerManager,
    "vllm": VllmServerManager,
}


__all__ = [
    "LlamaServerManager",
    "VllmServerManager",
    "servers_factory",
]
