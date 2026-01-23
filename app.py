from __future__ import annotations

import ast
import asyncio
import io
import os
import queue
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import Tracing components
from tracer import QueueSink, Tracer
from instrumentation import Instrumentor

# Import TinyTorch components
import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Dropout, Layer, Sequential
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, GELU, Softmax, LogSoftmax
from tinytorch.core.losses import MSELoss, CrossEntropyLoss, log_softmax
from tinytorch.core.norms import RMSNorm

# Import additional modules
from tinytorch.core.autograd import Function, enable_autograd
from tinytorch.core.optimizers import Optimizer, SGD, Adam, AdamW
from tinytorch.core.tokenization import Tokenizer, CharTokenizer, BPETokenizer, create_tokenizer, tokenize_dataset
from tinytorch.core.training import CosineSchedule, clip_grad_norm, Trainer
from tinytorch.core.embeddings import Embedding, PositionalEncoding, EmbeddingLayer, create_sinusoidal_embeddings

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AutoNameTransformer(ast.NodeTransformer):
    """
    AST transformer that automatically wraps assignments to Tensor-like values
    with a call to __auto_name__(name, value) so we can capture variable names.
    
    Transforms:
        x = Tensor([1,2,3])
    Into:
        x = __auto_name__("x", Tensor([1,2,3]))
    """
    
    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # Only handle simple single-target assignments like: x = ...
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            
            # Skip private/dunder names
            if var_name.startswith('_'):
                return node
            
            # Wrap the value in __auto_name__(name, value)
            new_value = ast.Call(
                func=ast.Name(id='__auto_name__', ctx=ast.Load()),
                args=[
                    ast.Constant(value=var_name),
                    node.value
                ],
                keywords=[]
            )
            
            # Create new assignment with wrapped value
            new_node = ast.Assign(
                targets=node.targets,
                value=new_value
            )
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node
        
        return node


def transform_code(code: str) -> str:
    """
    Transform user code to automatically capture variable names for Tensors.
    """
    try:
        tree = ast.parse(code)
        transformer = AutoNameTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except SyntaxError:
        # If parsing fails, return original code and let execution handle the error
        return code


def _make_exec_env(tracer: Tracer) -> Dict[str, Any]:
    """
    Execution environment for user-authored Python snippets.
    Provides direct access to TinyTorch classes and tracer utilities.
    """
    import builtins

    # Helper to allow users to manually box things
    def manual_box(label, tensors, scheme="1", parent=None):
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        tracer.box(label=label, tensors=tensors, scheme=str(scheme), parent_box=parent)

    # Auto-naming helper that gets injected into transformed code
    def auto_name(name: str, value: Any) -> Any:
        """Automatically names Tensor values when they're assigned to variables."""
        if isinstance(value, Tensor):
            tracer.name(value, name)
        return value

    # Start with a clean slate but include essential builtins
    env = {}

    # Manually add critical builtins
    env['__builtins__'] = builtins.__dict__
    env['__build_class__'] = builtins.__build_class__
    env['__name__'] = '__main__'
    env['__doc__'] = None

    # Add common builtins
    for name in ['print', 'len', 'range', 'int', 'float', 'str', 'list', 'dict', 'tuple',
                 'set', 'bool', 'type', 'isinstance', 'issubclass', 'super', 'object',
                 'Exception', 'ValueError', 'TypeError', 'AttributeError', 'KeyError',
                 'zip', 'enumerate', 'map', 'filter', 'sorted', 'reversed', 'abs',
                 'min', 'max', 'sum', 'round', 'pow', 'divmod', 'hash', 'id']:
        env[name] = getattr(builtins, name)

    # Add modules
    env['math'] = __import__('math')
    env['np'] = np
    env['numpy'] = np

    # Add TinyTorch components
    tiny_torch = {
        "Tensor": Tensor,
        "Linear": Linear,
        "Dropout": Dropout,
        "Sequential": Sequential,
        "Layer": Layer,
        "ReLU": ReLU,
        "Sigmoid": Sigmoid,
        "Tanh": Tanh,
        "GELU": GELU,
        "Softmax": Softmax,
        "LogSoftmax": LogSoftmax,
        "MSELoss": MSELoss,
        "CrossEntropyLoss": CrossEntropyLoss,
        "log_softmax": log_softmax,
        "RMSNorm": RMSNorm,
        "Function": Function,
        "enable_autograd": enable_autograd,
        "Optimizer": Optimizer,
        "SGD": SGD,
        "Adam": Adam,
        "AdamW": AdamW,
        "Tokenizer": Tokenizer,
        "CharTokenizer": CharTokenizer,
        "BPETokenizer": BPETokenizer,
        "create_tokenizer": create_tokenizer,
        "tokenize_dataset": tokenize_dataset,
        "CosineSchedule": CosineSchedule,
        "clip_grad_norm": clip_grad_norm,
        "Trainer": Trainer,
        "Embedding": Embedding,
        "PositionalEncoding": PositionalEncoding,
        "EmbeddingLayer": EmbeddingLayer,
        "create_sinusoidal_embeddings": create_sinusoidal_embeddings,
        "tracer": tracer,
        "box": manual_box,
        "__auto_name__": auto_name,
    }

    env.update(tiny_torch)

    return env


class PrintCapture(io.StringIO):
    """Captures print output and sends it to the tracer."""
    def __init__(self, tracer: Tracer):
        super().__init__()
        self.tracer = tracer
        
    def write(self, text: str) -> int:
        # Send non-empty text to tracer
        if text and text.strip():
            self.tracer.print(text.rstrip('\n'))
        return len(text)
    
    def flush(self):
        pass


def _run_user_code(code: str, tracer: Tracer) -> None:
    # TEMPORARY: Skip transformation to debug
    transformed_code = transform_code(code)  # Instead of transform_code(code)

    # 2. Setup Environment
    env = _make_exec_env(tracer)

    # 3. Redirect stdout to capture print statements
    old_stdout = sys.stdout
    sys.stdout = PrintCapture(tracer)

    # 4. Instrument Tensor/Layer classes to talk to our tracer
    with Instrumentor(tracer):
        try:
            # 5. Execute transformed code
            exec(transformed_code, env)
        except Exception:
            tracer.error(traceback.format_exc())
        finally:
            sys.stdout = old_stdout
            tracer.done()


async def _stream_queue_to_ws(ws: WebSocket, q: "queue.Queue[dict | None]") -> None:
    while True:
        item = await asyncio.to_thread(q.get)
        if item is None:
            return
        await ws.send_json(item)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            msg = await ws.receive_json()
            if not isinstance(msg, dict):
                continue

            action = msg.get("action")
            if action != "run":
                await ws.send_json({"event": "error", "message": "Unsupported action"})
                continue

            code = msg.get("code", "")
            q: "queue.Queue[dict | None]" = queue.Queue()
            tracer = Tracer(QueueSink(q))

            # Reset frontend state
            await ws.send_json({"event": "reset"})

            sender = asyncio.create_task(_stream_queue_to_ws(ws, q))

            # Run code in thread to avoid blocking async loop
            await asyncio.to_thread(_run_user_code, code, tracer)

            q.put(None)  # Signal end of stream
            await sender

    except WebSocketDisconnect:
        return


# Entry point for running the app
if __name__ == "__main__":
    import uvicorn
    
    # Support for Hugging Face Spaces (uses port 7860) and local development
    # HF Spaces sets SPACE_ID environment variable
    is_hf_space = os.environ.get("SPACE_ID") is not None
    
    # Get host and port from environment variables, with sensible defaults
    host = os.environ.get("HOST", "0.0.0.0" if is_hf_space else "127.0.0.1")
    port = int(os.environ.get("PORT", "7860" if is_hf_space else "8000"))
    
    print(f"Starting TinyTorch Visualizer on http://{host}:{port}")
    if is_hf_space:
        print("Running in Hugging Face Spaces mode")
    else:
        print("Running in local development mode")
        print(f"Open http://localhost:{port} in your browser")
    
    uvicorn.run(app, host=host, port=port)
