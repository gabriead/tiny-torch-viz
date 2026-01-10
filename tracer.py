# tracer.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, List
import numpy as np


@dataclass
class TraceEvent:
    event: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        d = {"event": self.event}
        d.update(self.payload)
        return d


class QueueSink:
    """
    Simple sink that puts already-JSON-serializable dicts onto a queue.
    Your app.py should have a background task reading this queue and sending
    each dict via ws.send_json(...).
    """
    def __init__(self, q):
        self.q = q

    def emit(self, ev: dict) -> None:
        self.q.put(ev)


class Tracer:
    """
    Events emitted:
      - tensor: {id, shape, size, dtype, nbytes, data, name?}
      - op:     {type, inputs:[ids], output:id, meta:{...}}  (meta includes memory stats)
      - box:    {label, tensors:[ids], scheme, parentBox?}
      - error:  {message}
      - done:   {}
    """

    def __init__(self, sink: QueueSink):
        self.sink = sink
        self._next_tid = 1
        self._next_cid = 1
        self._id_map: Dict[int, str] = {}   # id(obj)->tN
        self._names: Dict[str, str] = {}    # tN->name

    def _tid(self, obj: Any) -> str:
        oid = id(obj)
        tid = self._id_map.get(oid)
        if tid is None:
            tid = f"t{self._next_tid}"
            self._next_tid += 1
            self._id_map[oid] = tid
        return tid

    def _cid(self) -> str:
        cid = f"c{self._next_cid}"
        self._next_cid += 1
        return cid

    def name(self, t: Any, name: str) -> None:
        tid = self._tid(t)
        self._names[tid] = name
        # upsert tensor with name
        self.tensor(t)

    def _emit_tensor_payload(self, node_id: str, arr: np.ndarray, name: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {
            "id": node_id,
            "shape": list(arr.shape),
            "size": int(arr.size),
            "dtype": str(arr.dtype),
            "nbytes": int(arr.nbytes),
            "data": arr.tolist(),
        }
        if name:
            payload["name"] = name
        self.sink.emit(TraceEvent("tensor", payload).asdict())

    def tensor(self, obj: Any) -> str:
        """
        Upsert a tensor-like node. If obj has .data, treat it as a Tensor.
        Otherwise, treat obj as a constant/scalar and create a synthetic const node.
        """
        if hasattr(obj, "data"):
            tid = self._tid(obj)
            arr = np.asarray(obj.data)
            name = self._names.get(tid)
            self._emit_tensor_payload(tid, arr, name=name)
            return tid

        # Constant/scalar output: represent as a 0D or 1x1 tensor for visualization
        cid = self._cid()
        arr = np.asarray(obj, dtype=np.float32)
        self._emit_tensor_payload(cid, arr, name=None)
        return cid

    def op(self, op_type: str, inputs: Sequence[Any], output: Any, meta: Optional[dict] = None) -> None:
        meta = dict(meta or {})

        # Convert inputs/outputs into node ids (and emit tensor upserts)
        in_ids: List[str] = [self.tensor(x) for x in inputs]
        out_id: str = self.tensor(output)

        # Memory accounting
        def bytes_of(x: Any) -> int:
            if hasattr(x, "data"):
                return int(np.asarray(x.data).nbytes)
            return 0

        in_bytes_list = [bytes_of(x) for x in inputs]

        # output may be Tensor-like or constant
        if hasattr(output, "data"):
            out_bytes = int(np.asarray(output.data).nbytes)
        else:
            out_bytes = int(np.asarray(output, dtype=np.float32).nbytes)

        meta.setdefault("mem_in_bytes", int(sum(in_bytes_list)))
        meta.setdefault("mem_in_bytes_list", [int(b) for b in in_bytes_list])
        meta.setdefault("mem_out_bytes", int(out_bytes))
        # simple conservative upper bound for “peak” if all inputs + output coexist
        meta.setdefault("mem_peak_bytes", int(sum(in_bytes_list) + out_bytes))

        self.sink.emit(
            TraceEvent(
                "op",
                {
                    "type": op_type,
                    "inputs": in_ids,
                    "output": out_id,
                    "meta": meta,
                },
            ).asdict()
        )

    def box(
        self,
        label: str,
        tensors: Sequence[Any],
        scheme: str = "1",
        parent_box: Optional[str] = None,
    ) -> None:
        """
        Emits a 'box' event. The frontend expects:
          {event:'box', label, scheme, parentBox, tensors:[ids]}
        """
        t_ids: List[str] = []
        for t in tensors:
            # If it's a Tensor-like object, use stable tid mapping.
            # If it's a scalar/const, create a const node id.
            if hasattr(t, "data"):
                t_ids.append(self._tid(t))
                # Ensure tensor node exists/upserted (shape, data, etc.)
                self.tensor(t)
            else:
                t_ids.append(self.tensor(t))

        payload: Dict[str, Any] = {
            "label": label,
            "tensors": t_ids,
            "scheme": str(scheme),
        }
        if parent_box is not None:
            payload["parentBox"] = parent_box

        self.sink.emit(TraceEvent("box", payload).asdict())

    def error(self, message: str) -> None:
        self.sink.emit(TraceEvent("error", {"message": str(message)}).asdict())

    def done(self) -> None:
        self.sink.emit(TraceEvent("done", {}).asdict())
