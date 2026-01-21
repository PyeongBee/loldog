#!/usr/bin/env python3
"""
Generate synthetic YOLO training data by compositing champion images
onto minimap backgrounds with randomized placement and augmentations.

Output format:
  <output_dir>/
    images/000001.jpg
    labels/000001.txt   # YOLO label: class x_center y_center w h

Example:
  python generate_dataset.py ^
    --minimap_dir "C:/path/minimap" ^
    --champion_dir "C:/path/champions" ^
    --output_dir "C:/path/out" ^
    --count 2000 --img_size 640
"""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter


def list_images(directory: Path) -> List[Path]:
  exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
  return [p for p in directory.rglob("*") if p.suffix.lower() in exts]


def clamp(value: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, value))


def pil_to_rgb(img: Image.Image) -> Image.Image:
  if img.mode != "RGB":
    return img.convert("RGB")
  return img


def alpha_paste(bg: Image.Image, fg: Image.Image, pos: Tuple[int, int]) -> None:
  if fg.mode != "RGBA":
    fg = fg.convert("RGBA")
  bg.paste(fg, pos, fg.split()[3])


def resize_keep_aspect(img: Image.Image, target_long: int) -> Image.Image:
  w, h = img.size
  if w == 0 or h == 0:
    return img
  if w >= h:
    new_w = target_long
    new_h = int(h * (target_long / w))
  else:
    new_h = target_long
    new_w = int(w * (target_long / h))
  return img.resize((max(1, new_w), max(1, new_h)), Image.BICUBIC)


def add_moire(img: Image.Image, strength: float = 0.08) -> Image.Image:
  w, h = img.size
  freq = random.uniform(8.0, 16.0)
  angle = random.uniform(0, math.pi)
  x = np.linspace(0, 2 * math.pi * freq, w)
  y = np.linspace(0, 2 * math.pi * freq, h)
  xv, yv = np.meshgrid(x, y)
  pattern = np.sin(xv * math.cos(angle) + yv * math.sin(angle))
  pattern = (pattern * 0.5 + 0.5) * 255.0
  pat = Image.fromarray(pattern.astype(np.uint8), mode="L").resize((w, h))
  pat = Image.merge("RGB", (pat, pat, pat))
  return Image.blend(img, pat, alpha=clamp(strength, 0.0, 0.3))


def add_glare(img: Image.Image, strength: float = 0.35) -> Image.Image:
  w, h = img.size
  overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
  draw = ImageDraw.Draw(overlay)
  cx = random.randint(int(w * 0.1), int(w * 0.9))
  cy = random.randint(int(h * 0.1), int(h * 0.9))
  radius = random.randint(int(min(w, h) * 0.15), int(min(w, h) * 0.35))
  color = (255, 255, 255, int(255 * clamp(strength, 0.1, 0.6)))
  draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
  overlay = overlay.filter(ImageFilter.GaussianBlur(radius=radius * 0.35))
  base = img.convert("RGBA")
  out = Image.alpha_composite(base, overlay)
  return out.convert("RGB")


def random_perspective_points(w: int, h: int, max_shift: float) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
  src = [(0, 0), (w, 0), (w, h), (0, h)]
  dx = w * max_shift
  dy = h * max_shift
  dst = [
    (random.uniform(0, dx), random.uniform(0, dy)),
    (w - random.uniform(0, dx), random.uniform(0, dy)),
    (w - random.uniform(0, dx), h - random.uniform(0, dy)),
    (random.uniform(0, dx), h - random.uniform(0, dy)),
  ]
  return src, dst


def solve_homography(src: Iterable[Tuple[float, float]], dst: Iterable[Tuple[float, float]]) -> np.ndarray:
  src = list(src)
  dst = list(dst)
  if len(src) != 4 or len(dst) != 4:
    raise ValueError("Need 4 point pairs for homography.")
  a = []
  b = []
  for (x, y), (u, v) in zip(src, dst):
    a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
    a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b.append(u)
    b.append(v)
  a = np.array(a, dtype=np.float32)
  b = np.array(b, dtype=np.float32)
  h, *_ = np.linalg.lstsq(a, b, rcond=None)
  h = np.append(h, 1.0).reshape(3, 3)
  return h


def perspective_coeffs(dst: List[Tuple[float, float]], src: List[Tuple[float, float]]) -> List[float]:
  h = solve_homography(dst, src)
  return (h / h[2, 2]).reshape(-1)[:8].tolist()


def apply_homography(points: List[Tuple[float, float]], h: np.ndarray) -> List[Tuple[float, float]]:
  out = []
  for x, y in points:
    vec = np.array([x, y, 1.0], dtype=np.float32)
    tx, ty, tz = h @ vec
    if tz == 0:
      out.append((x, y))
    else:
      out.append((tx / tz, ty / tz))
  return out


def augment_image(
  img: Image.Image,
  bbox: Tuple[float, float, float, float],
  apply_perspective: bool,
  apply_moire_flag: bool,
  apply_glare_flag: bool,
) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
  w, h = img.size
  x1, y1, x2, y2 = bbox

  if apply_perspective:
    src, dst = random_perspective_points(w, h, max_shift=0.08)
    coeffs = perspective_coeffs(dst, src)
    img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    h_mat = solve_homography(src, dst)
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    tpts = apply_homography(pts, h_mat)
    xs = [p[0] for p in tpts]
    ys = [p[1] for p in tpts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

  if apply_moire_flag:
    img = add_moire(img, strength=random.uniform(0.05, 0.12))

  if apply_glare_flag:
    img = add_glare(img, strength=random.uniform(0.2, 0.5))

  if random.random() < 0.5:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
  if random.random() < 0.5:
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.2))

  return img, (x1, y1, x2, y2)


def generate_one(
  minimap_path: Path,
  champion_path: Path,
  img_size: int,
  scale_range: Tuple[float, float],
  apply_perspective: bool,
  apply_moire_flag: bool,
  apply_glare_flag: bool,
) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
  bg = pil_to_rgb(Image.open(minimap_path))
  if img_size > 0:
    bg = bg.resize((img_size, img_size), Image.BICUBIC)
  bw, bh = bg.size

  fg = Image.open(champion_path)
  if fg.mode != "RGBA":
    fg = fg.convert("RGBA")

  scale = random.uniform(*scale_range)
  target = int(min(bw, bh) * scale)
  fg = resize_keep_aspect(fg, max(8, target))
  fw, fh = fg.size

  max_x = max(1, bw - fw)
  max_y = max(1, bh - fh)
  x = random.randint(0, max_x)
  y = random.randint(0, max_y)

  alpha_paste(bg, fg, (x, y))
  bbox = (x, y, x + fw, y + fh)

  bg, bbox = augment_image(bg, bbox, apply_perspective, apply_moire_flag, apply_glare_flag)
  return bg, bbox


def save_yolo_label(path: Path, bbox: Tuple[float, float, float, float], img_size: int, class_id: int = 0) -> None:
  x1, y1, x2, y2 = bbox
  x1 = clamp(x1, 0, img_size)
  y1 = clamp(y1, 0, img_size)
  x2 = clamp(x2, 0, img_size)
  y2 = clamp(y2, 0, img_size)
  if x2 <= x1 or y2 <= y1:
    return
  cx = (x1 + x2) / 2 / img_size
  cy = (y1 + y2) / 2 / img_size
  w = (x2 - x1) / img_size
  h = (y2 - y1) / img_size
  path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Synthetic minimap dataset generator.")
  parser.add_argument("--minimap_dir", required=True, type=Path, help="Directory with minimap images.")
  parser.add_argument("--champion_dir", required=True, type=Path, help="Directory with champion images (PNG recommended).")
  parser.add_argument("--output_dir", required=True, type=Path, help="Output dataset directory.")
  parser.add_argument("--count", type=int, default=1000, help="Number of samples to generate.")
  parser.add_argument("--img_size", type=int, default=320, help="Output image size (square).")
  parser.add_argument("--seed", type=int, default=42, help="Random seed.")
  parser.add_argument("--min_scale", type=float, default=0.08, help="Min champion scale (relative to canvas).")
  parser.add_argument("--max_scale", type=float, default=0.2, help="Max champion scale (relative to canvas).")
  parser.add_argument("--moire_prob", type=float, default=0.6, help="Probability of moire augmentation.")
  parser.add_argument("--glare_prob", type=float, default=0.6, help="Probability of glare augmentation.")
  parser.add_argument("--perspective_prob", type=float, default=0.5, help="Probability of perspective warp.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  random.seed(args.seed)
  np.random.seed(args.seed)

  minimaps = list_images(args.minimap_dir)
  champions = list_images(args.champion_dir)
  if not minimaps:
    raise SystemExit(f"No minimap images in {args.minimap_dir}")
  if not champions:
    raise SystemExit(f"No champion images in {args.champion_dir}")

  images_dir = args.output_dir / "images"
  labels_dir = args.output_dir / "labels"
  images_dir.mkdir(parents=True, exist_ok=True)
  labels_dir.mkdir(parents=True, exist_ok=True)

  for i in range(1, args.count + 1):
    tries = 0
    while True:
      tries += 1
      if tries > 5:
        break
      bg_path = random.choice(minimaps)
      fg_path = random.choice(champions)
      apply_moire_flag = random.random() < args.moire_prob
      apply_glare_flag = random.random() < args.glare_prob
      apply_perspective = random.random() < args.perspective_prob
      img, bbox = generate_one(
        bg_path,
        fg_path,
        img_size=args.img_size,
        scale_range=(args.min_scale, args.max_scale),
        apply_perspective=apply_perspective,
        apply_moire_flag=apply_moire_flag,
        apply_glare_flag=apply_glare_flag,
      )
      x1, y1, x2, y2 = bbox
      if (x2 - x1) > 2 and (y2 - y1) > 2:
        break

    out_name = f"{i:06d}"
    img_path = images_dir / f"{out_name}.jpg"
    label_path = labels_dir / f"{out_name}.txt"
    img.save(img_path, quality=92)
    save_yolo_label(label_path, bbox, args.img_size, class_id=0)

  print(f"Done. images: {images_dir} labels: {labels_dir}")


if __name__ == "__main__":
  main()
