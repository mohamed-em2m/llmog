"""
`servers/__init__.py` — exports managers so the rest of the codebase
only ever imports from `servers`.
"""

# Import by name so clients can do:
#   from .servers import LlamaServerManager, VllmServerManager
from .llama_server_manager import LlamaServerManager
from .llama_cpp_python_manager import LlamaCppPythonManager
from .vllm_server_manager import VllmServerManager

servers_factory = {
    "llama_cpp": LlamaServerManager,            # native llama.cpp llama-server binary
    "llama_cpp_python": LlamaCppPythonManager,  # llama-cpp-python's built-in server
    "vllm": VllmServerManager,
}


__all__ = [
    "LlamaServerManager",
    "LlamaCppPythonManager",
    "VllmServerManager",
    "servers_factory",
]
