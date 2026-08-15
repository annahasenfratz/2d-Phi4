#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_ensemble_name(path: Path) -> tuple[float | None, int | None]:
    text = path.name
    km = re.search(r"kappac([0-9]+)p([0-9]+)", text)
    lm = re.search(r"_L([0-9]+)(?:_|$)", text)
    kappa = float(f"{km.group(1)}.{km.group(2)}") if km else None
    L = int(lm.group(1)) if lm else None
    return kappa, L


def load_configs(path: Path) -> np.ndarray:
    with np.load(path) as data:
        for key in ("phi", "configs", "arr_0"):
            if key in data.files:
                arr = data[key]
                break
        else:
            arr = data[data.files[0]]
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"expected configs with shape (N,L,L), got {arr.shape} in {path}")
    return arr


def binder_u4(phi: np.ndarray) -> float:
    m = phi.mean(axis=(1, 2))
    m2 = float(np.mean(m * m))
    m4 = float(np.mean(m**4))
    return float(1.0 - m4 / max(3.0 * m2 * m2, 1.0e-300))


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_plot_pdf(rows: list[dict[str, object]], out_pdf: Path, xlim: tuple[float, float] | None = None) -> None:
    scale = 3
    width, height = 720 * scale, 480 * scale
    margin_left, margin_right = 95 * scale, 35 * scale
    margin_top, margin_bottom = 45 * scale, 80 * scale
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title = font(18 * scale)
    f_axis = font(12 * scale)
    f_tick = font(10 * scale)

    plot_rows = [r for r in rows if r["kappa"] is not None]
    xs = [float(r["kappa"]) for r in plot_rows]
    ys = [float(r["Binder_U4"]) for r in plot_rows]
    xmin, xmax = xlim if xlim is not None else (min(xs), max(xs))
    ymin, ymax = min(ys), max(ys)
    xpad = 0.0 if xlim is not None else (0.04 * (xmax - xmin) if xmax > xmin else 0.01)
    ypad = 0.08 * (ymax - ymin) if ymax > ymin else 0.05
    xmin -= xpad
    xmax += xpad
    ymin = max(0.0, ymin - ypad)
    ymax = min(2.0 / 3.0, ymax + ypad)

    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top

    def px(x: float) -> int:
        return int(x0 + (x - xmin) / (xmax - xmin) * (x1 - x0))

    def py(y: float) -> int:
        return int(y0 - (y - ymin) / (ymax - ymin) * (y0 - y1))

    draw.text((width // 2, 16 * scale), "Binder cumulant by ensemble", fill="black", font=f_title, anchor="mt")
    draw.line([(x0, y0), (x1, y0)], fill="black", width=2 * scale)
    draw.line([(x0, y0), (x0, y1)], fill="black", width=2 * scale)

    for i in range(6):
        x = xmin + i * (xmax - xmin) / 5.0
        xp = px(x)
        draw.line([(xp, y0), (xp, y0 + 5 * scale)], fill="black", width=scale)
        draw.text((xp, y0 + 10 * scale), f"{x:.3f}", fill="black", font=f_tick, anchor="mt")
    for i in range(6):
        y = ymin + i * (ymax - ymin) / 5.0
        yp = py(y)
        draw.line([(x0 - 5 * scale, yp), (x0, yp)], fill="black", width=scale)
        draw.line([(x0, yp), (x1, yp)], fill=(225, 225, 225), width=scale)
        draw.text((x0 - 10 * scale, yp), f"{y:.2f}", fill="black", font=f_tick, anchor="rm")

    draw.text(((x0 + x1) // 2, height - 30 * scale), "kappa", fill="black", font=f_axis, anchor="mm")
    draw.text((30 * scale, (y0 + y1) // 2), "U4", fill="black", font=f_axis, anchor="mm")

    colors = {
        8: (39, 120, 181),
        16: (217, 95, 2),
        32: (27, 158, 119),
        64: (117, 112, 179),
    }
    legend_x = x1 - 80 * scale
    legend_y = y1 + 12 * scale
    for idx, L in enumerate(sorted({int(r["L"]) for r in plot_rows})):
        subset = [r for r in plot_rows if int(r["L"]) == L]
        subset.sort(key=lambda r: float(r["kappa"]))
        color = colors.get(L, (90, 90, 90))
        points = [(px(float(r["kappa"])), py(float(r["Binder_U4"]))) for r in subset]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3 * scale)
        radius = 4 * scale
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black", width=scale)
        ly = legend_y + idx * 20 * scale
        draw.line([(legend_x, ly), (legend_x + 24 * scale, ly)], fill=color, width=3 * scale)
        draw.text((legend_x + 30 * scale, ly), f"L={L}", fill="black", font=f_tick, anchor="lm")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_pdf, "PDF", resolution=300.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Binder_U4 from all configs.npz ensembles into a kappa/L table.")
    ap.add_argument("root", type=Path, help="Directory containing ensemble subdirectories.")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-pdf", type=Path, default=None)
    ap.add_argument("--kappa-min", type=float, default=None)
    ap.add_argument("--kappa-max", type=float, default=None)
    args = ap.parse_args()

    root = args.root
    out_csv = args.out_csv or (root / "binder_u4_kappa_L_table.csv")
    out_pdf = args.out_pdf or (root / "binder_u4_vs_kappa_by_L.pdf")

    rows: list[dict[str, object]] = []
    for cfg in sorted(root.glob("*/configs.npz")):
        ensemble = cfg.parent.name
        kappa, L_from_name = parse_ensemble_name(cfg.parent)
        phi = load_configs(cfg)
        n, L, _ = phi.shape
        if L_from_name is not None and L_from_name != L:
            raise ValueError(f"L mismatch for {cfg}: name has L={L_from_name}, configs have L={L}")
        rows.append(
            {
                "ensemble": ensemble,
                "kappa": kappa,
                "L": L,
                "N": n,
                "Binder_U4": binder_u4(phi),
                "configs": str(cfg),
            }
        )

    rows.sort(key=lambda r: (int(r["L"]), float(r["kappa"]) if r["kappa"] is not None else -1.0, str(r["ensemble"])))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ensemble", "kappa", "L", "N", "Binder_U4", "configs"])
        writer.writeheader()
        writer.writerows(rows)

    xlim = None
    if args.kappa_min is not None or args.kappa_max is not None:
        if args.kappa_min is None or args.kappa_max is None:
            raise SystemExit("--kappa-min and --kappa-max must be supplied together")
        if args.kappa_min >= args.kappa_max:
            raise SystemExit("--kappa-min must be smaller than --kappa-max")
        xlim = (args.kappa_min, args.kappa_max)
    draw_plot_pdf(rows, out_pdf, xlim=xlim)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Wrote plot to {out_pdf}")


if __name__ == "__main__":
    main()
