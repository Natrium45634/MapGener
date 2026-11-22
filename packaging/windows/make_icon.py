"""Draw the HexWorld icon (a globe-coloured hexagon) and write ``hexworld.ico`` + PNG.

No dependencies beyond numpy: the ICO holds PNG images of 16…256 px.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hexworld.render import write_png  # noqa: E402


def draw(size: int) -> np.ndarray:
    s = 4  # supersampling
    n = size * s
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n - 1) / 2.0
    r = 0.47 * n
    dx, dy = np.abs(x - cx), np.abs(y - cy)
    # flat-top hexagon
    inside = (dy <= r * np.sqrt(3) / 2) & (np.sqrt(3) * dx + dy <= np.sqrt(3) * r)
    edge = inside & ~(
        (dy <= 0.9 * r * np.sqrt(3) / 2) & (np.sqrt(3) * dx + dy <= np.sqrt(3) * 0.9 * r)
    )
    rgba = np.zeros((n, n, 4))
    # sea, two continents, ice cap
    sea = np.array([38, 92, 168], float)
    land = np.array([96, 150, 72], float)
    desert = np.array([214, 184, 112], float)
    ice = np.array([236, 242, 248], float)
    u, v = (x - cx) / r, (y - cy) / r
    blob1 = ((u + 0.25) / 0.45) ** 2 + ((v - 0.05) / 0.6) ** 2 < 1
    blob2 = ((u - 0.38) / 0.3) ** 2 + ((v + 0.25) / 0.28) ** 2 < 1
    col = np.where(blob1[..., None] | blob2[..., None], land, sea)
    col = np.where((blob1 & (v > 0.15))[..., None], desert, col)
    col = np.where((v < -0.62)[..., None], ice, col)
    shade = 1.0 - 0.25 * np.clip((u + v) / 2.0, -1, 1)
    col = col * shade[..., None]
    col = np.where(edge[..., None], np.array([28, 36, 52], float), col)
    rgba[..., :3] = col
    rgba[..., 3] = np.where(inside, 255.0, 0.0)
    small = rgba.reshape(size, s, size, s, 4).mean(axis=(1, 3))
    return np.clip(np.rint(small), 0, 255).astype(np.uint8)


def write_rgba_png(path: Path, rgba: np.ndarray) -> bytes:
    import zlib

    h, w = rgba.shape[:2]
    raw = np.empty((h, 1 + 4 * w), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = rgba.reshape(h, 4 * w)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw.tobytes(), 9))
        + chunk(b"IEND", b"")
    )
    if path is not None:
        path.write_bytes(data)
    return data


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = (16, 24, 32, 48, 64, 128, 256)
    images = [write_rgba_png(None, draw(sz)) for sz in sizes]
    header = struct.pack("<HHH", 0, 1, len(sizes))
    entries, offset = b"", 6 + 16 * len(sizes)
    for sz, data in zip(sizes, images, strict=True):
        dim = 0 if sz >= 256 else sz
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    (out / "hexworld.ico").write_bytes(header + entries + b"".join(images))
    write_rgba_png(out / "hexworld.png", draw(256))
    write_png(out / "hexworld_preview.png", draw(128)[..., :3])
    write_version_info(out / "version_info.txt")
    print(out / "hexworld.ico")


def write_version_info(path: Path) -> None:
    """Windows file properties of HexWorld.exe (PyInstaller version resource)."""
    from hexworld.version import PRODUCT_VERSION

    nums = [int(x) for x in PRODUCT_VERSION.split(".")] + [0] * 4
    v = tuple(nums[:4])
    path.write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={v}, prodvers={v}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('041904B0', [
      StringStruct('CompanyName', 'HexWorld'),
      StringStruct('FileDescription', 'HexWorld — генератор карт миров'),
      StringStruct('FileVersion', '{PRODUCT_VERSION}'),
      StringStruct('InternalName', 'HexWorld'),
      StringStruct('OriginalFilename', 'HexWorld.exe'),
      StringStruct('ProductName', 'HexWorld'),
      StringStruct('ProductVersion', '{PRODUCT_VERSION}')])]),
    VarFileInfo([VarStruct('Translation', [0x0419, 1200])])
  ]
)
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "packaging/windows/build")
