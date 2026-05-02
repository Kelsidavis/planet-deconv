"""Unified video reader: dispatches SER and AVI to a common interface."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ser import SerFile


class VideoFile(Protocol):
    width: int
    height: int

    def __len__(self) -> int: ...
    def read_frame(self, idx: int) -> np.ndarray: ...
    def close(self) -> None: ...


class _SerWrapper:
    def __init__(self, path: Path):
        self._ser = SerFile(path)
        self.width = self._ser.header.width
        self.height = self._ser.header.height

    def __len__(self) -> int:
        return len(self._ser)

    def read_frame(self, idx: int) -> np.ndarray:
        return self._ser.read_frame(idx)

    def close(self) -> None:
        self._ser.close()


class _AviWrapper:
    """Sequential-only AVI reader via imageio + pyav.

    cv2 segfaults on some uncompressed planetary AVIs; pyav handles them.
    We expose forward-only iteration: callers iterating in order get full
    speed, random access rewinds and re-reads.
    """

    def __init__(self, path: Path):
        import imageio.v3 as iio  # lazy import

        self._iio = iio
        self._path = path
        meta = iio.immeta(str(path), plugin="pyav")
        # pyav-reported nframes can be wrong/missing; probe by iterating once.
        self._n = self._count_frames(path)
        first = next(iio.imiter(str(path), plugin="pyav"))
        self.height, self.width = first.shape[:2]
        self._iter = None
        self._next_idx = 0
        self._meta = meta
        self._open_iter()

    @staticmethod
    def _count_frames(path: Path) -> int:
        import imageio.v3 as iio

        n = 0
        for _ in iio.imiter(str(path), plugin="pyav"):
            n += 1
        return n

    def __len__(self) -> int:
        return self._n

    def _open_iter(self) -> None:
        self._iter = self._iio.imiter(str(self._path), plugin="pyav")
        self._next_idx = 0

    def read_frame(self, idx: int) -> np.ndarray:
        if idx < self._next_idx:
            self._open_iter()
        while self._next_idx < idx:
            try:
                next(self._iter)
            except StopIteration as e:
                raise IOError(f"frame {self._next_idx} past end of {self._path}") from e
            self._next_idx += 1
        try:
            frame = next(self._iter)
        except StopIteration as e:
            raise IOError(f"frame {idx} past end of {self._path}") from e
        self._next_idx += 1
        if frame.ndim == 3 and frame.shape[2] >= 3:
            frame = frame[..., :3].mean(axis=-1).astype(np.uint8)
        return frame

    def close(self) -> None:
        self._iter = None


def open_video(path: str | Path) -> VideoFile:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".ser":
        return _SerWrapper(p)
    if suf == ".avi":
        return _AviWrapper(p)
    raise ValueError(f"unsupported video extension: {suf}")
