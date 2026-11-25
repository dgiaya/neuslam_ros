"""The thread that owns NeuSLAM: frames in from a bounded queue, outputs out as they arrive."""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional

from neuslam_ros.core.pipeline import Pipeline, Prepared


class SlamRunner:
    """`make_slam()` returns an unstarted `neuslam.stream.StreamingSLAM`. Frames offered while
    the queue is full (or before NeuSLAM is up) are dropped and counted; `block=True` waits instead."""

    def __init__(self, make_slam: Callable[[], Any], pipeline: Pipeline, *, on_output: Callable,
                 on_map: Optional[Callable] = None, on_error: Optional[Callable] = None,
                 max_queue: int = 2, log: Callable[[str], None] = print) -> None:
        self.make_slam, self.pipeline = make_slam, pipeline
        self.on_output, self.on_map, self.on_error, self.log = on_output, on_map, on_error, log
        self.frames: "queue.Queue[Prepared]" = queue.Queue(maxsize=max(1, int(max_queue)))
        self.calls: "queue.Queue[tuple]" = queue.Queue()
        self.ready = threading.Event()
        self.done = threading.Event()
        self._stop = threading.Event()
        self._save_to: Optional[str] = None
        self.dropped = 0
        self.slam: Any = None
        self._thread = threading.Thread(target=self._run, name="neuslam", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def offer(self, prep: Prepared, *, block: bool = False) -> bool:
        if not block and not self.ready.is_set():
            self.dropped += 1
            return False
        try:
            self.frames.put(prep, block=block, timeout=None)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def call(self, fn: Callable[[Any], Any]) -> Future:
        """Run `fn(slam)` on the NeuSLAM thread, between frames."""
        f: Future = Future()
        self.calls.put((fn, f))
        return f

    def stop(self, *, save_to: Optional[str] = None, timeout: Optional[float] = None) -> None:
        """Finish the frames already queued, end the stream (saving to `save_to`) and stop."""
        self._save_to = save_to
        self._stop.set()
        self._thread.join(timeout)

    def reset(self) -> Future:
        def do(_slam):
            self.slam.close()
            while not self.frames.empty():
                self.frames.get_nowait()
            self.pipeline.reset()
            self.slam = self.make_slam()
            self.slam.start()
        return self.call(do)

    def _emit(self) -> None:
        for r in self.slam.poll():
            out = self.pipeline.output(r)
            if out is not None:
                self.on_output(out)
        if self.on_map is not None:
            m = self.slam.take_map()
            if m is not None:
                self.on_map(m)

    def _run(self) -> None:
        try:
            self.slam = self.make_slam()
            self.slam.start()
            self.ready.set()
            self.log("NeuSLAM ready")
            current: Optional[Prepared] = None
            while True:
                while not self.calls.empty():
                    fn, fut = self.calls.get_nowait()
                    try:
                        fut.set_result(fn(self.slam))
                    except Exception as e:   # reported to the caller
                        fut.set_exception(e)
                if current is None:
                    try:
                        current = self.frames.get(timeout=0.005)
                    except queue.Empty:
                        if self._stop.is_set():
                            break
                if current is not None and self.pipeline.submit(self.slam, current, timeout=0.005):
                    current = None
                self._emit()
                if not self.slam.alive():
                    raise RuntimeError(f"NeuSLAM worker died: {self.slam.dead_workers()}")
            outputs = self.slam.finish(save_to=self._save_to)
            self._emit()
            if self._save_to:
                self.log(f"NeuSLAM outputs: {', '.join(sorted(outputs)) or 'none'} in {self._save_to}")
        except Exception as e:
            self.log(f"NeuSLAM stopped: {e!r}")
            if self.on_error is not None:
                self.on_error(e)
        finally:
            self.ready.clear()
            if self.slam is not None:
                self.slam.close()
            self.done.set()
