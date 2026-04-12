from __future__ import annotations

from queue import SimpleQueue


class LocalChannel:
    def __init__(self) -> None:
        self._queue: SimpleQueue[object] = SimpleQueue()

    def send(self, message: object) -> None:
        self._queue.put(message)

    def recv(self) -> object:
        return self._queue.get()
