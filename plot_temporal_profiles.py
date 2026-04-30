#!/usr/bin/env python3
"""Plot the four temporal BBE profiles used by the OTN simulator.

Mirrors _apply_profile_to_bbe() in otn_simulator.py so the shapes are
faithful to the real generator. Output: a single 1x4 figure suitable
for the "Failure Types and Metric Profiles" slide.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

T = 900
T0 = 120  # failure start
T1 = 820  # failure end
BASE_SPIKE = 10.0


def _apply_profile(series, t0, t1, base_spike, profile, rng):
    kind = profile["kind"]
    mult = float(profile["mult"])
    bursts = int(profile.get("bursts", 1))
    ramp_down = bool(profile.get("ramp_down", False))
    width = max(1, t1 - t0)

    if kind == "step":
        series[t0:t1] = np.maximum(series[t0:t1], base_spike * mult)

    elif kind == "ramp":
        ramp = np.linspace(0.2, 1.0, width, dtype=np.float32)
        if ramp_down:
            ramp = np.linspace(1.0, 0.2, width, dtype=np.float32)
        series[t0:t1] = np.maximum(series[t0:t1], base_spike * mult * ramp)

    elif kind == "burst":
        for _ in range(max(1, bursts)):
            b_w = max(1, int(width * rng.uniform(0.05, 0.15)))
            b0 = int(rng.integers(t0, max(t0 + 1, t1 - b_w + 1)))
            b1 = min(t1, b0 + b_w)
            series[b0:b1] = np.maximum(series[b0:b1], base_spike * mult)
        if ramp_down:
            tail0 = max(t0, t1 - max(1, int(width * 0.25)))
            tail = np.linspace(1.0, 0.3, t1 - tail0, dtype=np.float32)
            series[tail0:t1] = np.maximum(series[tail0:t1], base_spike * mult * tail)

    elif kind == "step_recovery":
        series[t0:t1] = np.maximum(series[t0:t1], base_spike * mult)
        for _ in range(max(1, bursts)):
            b_w = max(1, int(width * rng.uniform(0.03, 0.08)))
            b0 = int(rng.integers(t0, max(t0 + 1, t1 - b_w + 1)))
            b1 = min(t1, b0 + b_w)
            series[b0:b1] = np.maximum(series[b0:b1], base_spike * (mult * 1.2))
        if ramp_down:
            tail_factor = float(profile.get("tail_factor", 0.5))
            tail_len = max(1, int(width * tail_factor))
            tail0 = t1
            tail1 = min(len(series), t1 + tail_len)
            if tail1 > tail0:
                tail = np.linspace(1.0, 0.1, tail1 - tail0, dtype=np.float32)
                series[tail0:tail1] = np.maximum(series[tail0:tail1], base_spike * mult * tail)


PROFILES = [
    {
        "name": "Step",
        "subtitle": "Fiber Cut / XCON Port Down",
        "profile": {"kind": "step", "mult": 5.5, "bursts": 1, "ramp_down": False},
        "color": "#7B2CBF",
    },
    {
        "name": "Step Recovery",
        "subtitle": "Fiber Crack / Fabric Fault / Line Disconnect",
        "profile": {"kind": "step_recovery", "mult": 3.5, "bursts": 6,
                    "ramp_down": True, "tail_factor": 0.15},
        "color": "#9D4EDD",
    },
    {
        "name": "Ramp",
        "subtitle": "Fiber Aging",
        "profile": {"kind": "ramp", "mult": 2.8, "bursts": 1, "ramp_down": False},
        "color": "#5A189A",
    },
    {
        "name": "Burst",
        "subtitle": "XCON Buffer Overflow",
        "profile": {"kind": "burst", "mult": 4.6, "bursts": 8, "ramp_down": False},
        "color": "#C77DFF",
    },
]


def main() -> None:
    fig, axes_2d = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    axes = axes_2d.flatten()

    for ax, spec in zip(axes, PROFILES):
        rng = np.random.default_rng(7)  # reproducible burst positions
        series = np.zeros(T, dtype=np.float32)
        # add faint baseline noise for realism
        series += rng.normal(0.0, 0.25, size=T).clip(min=0.0)

        _apply_profile(series, T0, T1, BASE_SPIKE, spec["profile"], rng)

        t = np.arange(T)
        ax.fill_between(t, 0, series, color=spec["color"], alpha=0.25)
        ax.plot(t, series, color=spec["color"], linewidth=1.8)

        # Mark failure window
        ax.axvline(T0, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.axvline(T1, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)

        ax.set_title(spec["name"], fontsize=15, fontweight="bold",
                     color=spec["color"], pad=8)
        ax.text(0.5, -0.18, spec["subtitle"], transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="#444",
                style="italic")

        ax.set_xlabel("time (s)", fontsize=10)
        ax.set_xlim(0, T)
        ax.set_ylim(0, 65)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes_2d[0, 0].set_ylabel("BBE", fontsize=11, fontweight="bold")
    axes_2d[1, 0].set_ylabel("BBE", fontsize=11, fontweight="bold")

    fig.suptitle("Four Temporal BBE Profiles", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_png = "outputs/temporal_profiles.png"
    out_pdf = "outputs/temporal_profiles.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[ok] saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
