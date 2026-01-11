"""
Instrumentation module for tracing TinyTorch operations.

This module provides hooks into Tensor and Layer classes to automatically
emit trace events for visualization.
"""

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Layer
from tinytorch.core.activations import ReLU, Sigmoid, Tanh, GELU, Softmax
from tinytorch.core.losses import MSELoss, CrossEntropyLoss


class Instrumentor:
    """
    Context manager that instruments TinyTorch classes to emit trace events.
    
    Usage:
        tracer = Tracer(sink)
        with Instrumentor(tracer):
            # Tensor operations here will be traced
            a = Tensor([1, 2, 3])
            b = a + Tensor([4, 5, 6])
    """
    
    def __init__(self, tracer):
        self.tracer = tracer
        self.original_methods = {}
        self._inside_layer = False  # Flag to suppress internal op tracing

    def _wrap_tensor_method(self, method_name, op_name=None):
        """Wraps a Tensor method to emit an 'op' event."""
        if not hasattr(Tensor, method_name):
            return

        original = getattr(Tensor, method_name)
        self.original_methods[f"Tensor.{method_name}"] = original
        instrumentor = self  # Capture reference for closure

        op_type = op_name if op_name else method_name.strip('_')

        def wrapped(instance, *args, **kwargs):
            # Execute original logic
            result = original(instance, *args, **kwargs)

            # Skip tracing if we're inside a layer/activation call
            if instrumentor._inside_layer:
                return result

            # Build inputs list
            inputs = [instance]
            
            # Add tensor arguments to inputs
            for arg in args:
                if isinstance(arg, Tensor):
                    inputs.append(arg)
            
            # Build metadata from kwargs
            meta = {}
            for key, val in kwargs.items():
                if not isinstance(val, Tensor):
                    meta[key] = val

            # Emit op event
            instrumentor.tracer.op(op_type, inputs, result, meta)
            return result

        setattr(Tensor, method_name, wrapped)

    def _wrap_tensor_unary_method(self, method_name, op_name=None):
        """Wraps a Tensor unary method (no additional tensor args)."""
        if not hasattr(Tensor, method_name):
            return

        original = getattr(Tensor, method_name)
        self.original_methods[f"Tensor.{method_name}"] = original
        instrumentor = self  # Capture reference for closure

        op_type = op_name if op_name else method_name

        def wrapped(instance, *args, **kwargs):
            # Execute original logic
            result = original(instance, *args, **kwargs)

            # Skip tracing if we're inside a layer/activation call
            if instrumentor._inside_layer:
                return result

            # Build metadata from args/kwargs
            meta = {}
            for i, arg in enumerate(args):
                if not isinstance(arg, Tensor):
                    meta[f'arg{i}'] = arg
            for key, val in kwargs.items():
                if not isinstance(val, Tensor):
                    meta[key] = val

            # Emit op event
            instrumentor.tracer.op(op_type, [instance], result, meta)
            return result

        setattr(Tensor, method_name, wrapped)

    def _wrap_layer_forward(self):
        """Wraps Layer.__call__ to emit trace events for layer operations."""
        if not hasattr(Layer, '__call__'):
            return
            
        original = Layer.__call__
        self.original_methods["Layer.__call__"] = original
        instrumentor = self  # Capture reference for closure

        def wrapped(instance, x, *args, **kwargs):
            # Set flag to suppress internal tensor op tracing
            was_inside = instrumentor._inside_layer
            instrumentor._inside_layer = True
            try:
                # Execute original logic
                result = original(instance, x, *args, **kwargs)
            finally:
                instrumentor._inside_layer = was_inside

            # Get the layer name
            layer_name = instance.__class__.__name__

            # Build inputs list - for Linear, include weight and bias
            inputs = [x]
            meta = {'layer_type': layer_name}
            
            # For Linear layers, include weight and bias for visualization
            if layer_name == 'Linear':
                if hasattr(instance, 'weight'):
                    inputs.append(instance.weight)
                    meta['has_weight'] = True
                if hasattr(instance, 'bias') and instance.bias is not None:
                    inputs.append(instance.bias)
                    meta['has_bias'] = True

            # Emit op event for the layer
            instrumentor.tracer.op(layer_name.lower(), inputs, result, meta)

            return result

        setattr(Layer, "__call__", wrapped)

    def _wrap_activation(self, activation_cls):
        """Wraps an activation class's __call__ or forward method."""
        cls_name = activation_cls.__name__
        instrumentor = self  # Capture reference for closure
        
        # Check if it has __call__ method
        if hasattr(activation_cls, '__call__'):
            original = activation_cls.__call__
            self.original_methods[f"{cls_name}.__call__"] = original
            
            def make_wrapped(orig, name):
                def wrapped(instance, x, *args, **kwargs):
                    # Set flag to suppress internal tensor op tracing
                    was_inside = instrumentor._inside_layer
                    instrumentor._inside_layer = True
                    try:
                        result = orig(instance, x, *args, **kwargs)
                    finally:
                        instrumentor._inside_layer = was_inside
                    
                    meta = {'activation_type': name}
                    instrumentor.tracer.op(name.lower(), [x], result, meta)
                    return result
                return wrapped
            
            setattr(activation_cls, "__call__", make_wrapped(original, cls_name))

    def _wrap_loss(self, loss_cls):
        """Wraps a loss class's __call__ method."""
        cls_name = loss_cls.__name__
        instrumentor = self  # Capture reference for closure
        
        if hasattr(loss_cls, '__call__'):
            original = loss_cls.__call__
            self.original_methods[f"{cls_name}.__call__"] = original
            
            def make_wrapped(orig, name):
                def wrapped(instance, predictions, targets, *args, **kwargs):
                    # Set flag to suppress internal tensor op tracing
                    was_inside = instrumentor._inside_layer
                    instrumentor._inside_layer = True
                    try:
                        result = orig(instance, predictions, targets, *args, **kwargs)
                    finally:
                        instrumentor._inside_layer = was_inside
                    
                    meta = {'loss_type': name}
                    instrumentor.tracer.op(name.lower(), [predictions, targets], result, meta)
                    return result
                return wrapped
            
            setattr(loss_cls, "__call__", make_wrapped(original, cls_name))

    def instrument(self):
        """Apply all hooks."""
        # Binary arithmetic operations (use dunder methods to avoid double-tracing)
        self._wrap_tensor_method("__add__", "add")
        self._wrap_tensor_method("__sub__", "sub")
        self._wrap_tensor_method("__mul__", "mul")
        self._wrap_tensor_method("__truediv__", "div")
        # Wrap matmul (not __matmul__) since __matmul__ calls matmul internally
        # This way both a.matmul(b) and a @ b get traced (via the matmul call)
        self._wrap_tensor_method("matmul", "matmul")

        # Unary/transformation operations
        self._wrap_tensor_unary_method("transpose", "transpose")
        self._wrap_tensor_unary_method("reshape", "reshape")
        self._wrap_tensor_unary_method("sum", "sum")
        self._wrap_tensor_unary_method("mean", "mean")
        self._wrap_tensor_unary_method("max", "max")

        # Layers - Layer.__call__ handles Linear, Dropout, etc.
        self._wrap_layer_forward()
        
        # Activations - these don't inherit from Layer
        for activation_cls in [ReLU, Sigmoid, Tanh, GELU, Softmax]:
            self._wrap_activation(activation_cls)
        
        # Losses
        for loss_cls in [MSELoss, CrossEntropyLoss]:
            self._wrap_loss(loss_cls)

    def uninstrument(self):
        """Restore original methods."""
        # Map class names to actual classes
        cls_map = {
            "Tensor": Tensor,
            "Layer": Layer,
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
            "Tanh": Tanh,
            "GELU": GELU,
            "Softmax": Softmax,
            "MSELoss": MSELoss,
            "CrossEntropyLoss": CrossEntropyLoss,
        }
        
        for name, original in self.original_methods.items():
            parts = name.split('.')
            if len(parts) != 2:
                continue
            cls_name, method = parts
            
            if cls_name in cls_map:
                setattr(cls_map[cls_name], method, original)
                    
        self.original_methods.clear()

    def __enter__(self):
        self.instrument()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninstrument()
