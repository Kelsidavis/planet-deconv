"""SER planetary-capture reader.

178-byte header + raw frame data (+ optional 8-byte-per-frame timestamp trailer).
Spec: http://www.grischa-hahn.homepage.t-online.de/astro/ser/
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLOR_MONO = 0
COLOR_BAYER_RGGB = 8
COLOR_BAYER_GRBG = 9
COLOR_BAYER_GBRG = 10
COLOR_BAYER_BGGR = 11
COLOR_RGB = 100
COLOR_BGR = 101

_HEADER_FMT = "<14s i I I I I I I 40s 40s 40s q q"
_HEADER_SIZE = 178


@dataclass(frozen=True)
class SerHeader:
    file_id: str
    lu_id: int
    color_id: int
    little_endian: int
    width: int
    height: int
    bit_depth: int
    frame_count: int
    observer: str
    instrument: str
    telescope: str
    date_time: int
    date_time_utc: int

    @property
    def planes(self) -> int:
        return 3 if self.color_id in (COLOR_RGB, COLOR_BGR) else 1

    @property
    def bytes_per_pixel(self) -> int:
        return self.planes * (1 if self.bit_depth <= 8 else 2)

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * self.bytes_per_pixel

    @property
    def is_mono(self) -> bool:
        return self.color_id == COLOR_MONO

    @property
    def is_bayer(self) -> bool:
        return COLOR_BAYER_RGGB <= self.color_id <= COLOR_BAYER_BGGR


def _read_header(fh) -> SerHeader:
    raw = fh.read(_HEADER_SIZE)
    if len(raw) != _HEADER_SIZE:
        raise ValueError(f"file too short for SER header ({len(raw)} bytes)")
    f = struct.unpack(_HEADER_FMT, raw)
    return SerHeader(
        file_id=f[0].decode("ascii", errors="replace"),
        lu_id=f[1],
        color_id=f[2],
        little_endian=f[3],
        width=f[4],
        height=f[5],
        bit_depth=f[6],
        frame_count=f[7],
        observer=f[8].rstrip(b"\x00").decode("ascii", errors="replace"),
        instrument=f[9].rstrip(b"\x00").decode("ascii", errors="replace"),
        telescope=f[10].rstrip(b"\x00").decode("ascii", errors="replace"),
        date_time=f[11],
        date_time_utc=f[12],
    )


class SerFile:
    """Random-access reader for one SER capture."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = self.path.open("rb")
        self.header = _read_header(self._fh)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "SerFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return self.header.frame_count

    def read_frame(self, idx: int) -> np.ndarray:
        if not 0 <= idx < self.header.frame_count:
            raise IndexError(idx)
        h = self.header
        offset = _HEADER_SIZE + idx * h.frame_bytes
        self._fh.seek(offset)
        raw = self._fh.read(h.frame_bytes)
        if len(raw) != h.frame_bytes:
            raise IOError(f"short read at frame {idx}: got {len(raw)} of {h.frame_bytes}")
        if h.bit_depth <= 8:
            arr = np.frombuffer(raw, dtype=np.uint8)
        else:
            dt = "<u2" if h.little_endian else ">u2"
            arr = np.frombuffer(raw, dtype=dt)
        if h.planes == 3:
            arr = arr.reshape(h.height, h.width, 3)
            if h.color_id == COLOR_BGR:
                arr = arr[..., ::-1]
        else:
            arr = arr.reshape(h.height, h.width)
        return arr

    def iter_frames(self, start: int = 0, stop: int | None = None, step: int = 1):
        stop = self.header.frame_count if stop is None else stop
        for i in range(start, stop, step):
            yield i, self.read_frame(i)
