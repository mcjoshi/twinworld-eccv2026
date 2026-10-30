"""Find the best measured TUM/Gold Coast voxel allocation per point budget."""

from __future__ import annotations

import json
from pathlib import Path


TUM_INPUT = Path("experiments/2026-08-13-budget-allocation/tum.json")
GC_INPUT = Path("experiments/2026-08-13-gc-voxel-achieved/results.json")
OUTPUT = Path("experiments/2026-08-20-budget-curve/results.json")
BUDGETS = (40_000_000, 55_000_000, 70_000_000, 85_000_000, 100_000_000)
TUM_FINAL_SCENE_SCALE = 5 / 4


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required measured-curve file is missing: {path}")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return rows


def main() -> None:
    tum_rows = load_rows(TUM_INPUT)
    gc_rows = load_rows(GC_INPUT)

    candidates = [
        {
            "tum_voxel": tum["voxel"],
            "gc_voxel": gc["voxel"],
            "tum_points_final_phase": tum["points_4scenes"] * TUM_FINAL_SCENE_SCALE,
            "gc_points_final_phase": gc["points_2scenes"],
            "predicted_geometry": tum["mean_fscore"],
            "predicted_semantics": gc["loso_miou"],
        }
        for tum in tum_rows
        for gc in gc_rows
    ]
    for candidate in candidates:
        candidate["total_points_final_phase"] = (
            candidate["tum_points_final_phase"] + candidate["gc_points_final_phase"]
        )
        candidate["predicted_total"] = (
            candidate["predicted_geometry"] + candidate["predicted_semantics"]
        )

    results = []
    for budget in BUDGETS:
        feasible = [
            candidate
            for candidate in candidates
            if candidate["total_points_final_phase"] <= budget
        ]
        if not feasible:
            results.append(
                {
                    "budget_points": budget,
                    "feasible": False,
                    "reason": "No measured TUM/Gold Coast voxel combination fits this budget.",
                }
            )
            continue

        best = max(
            feasible,
            key=lambda candidate: (
                candidate["predicted_total"],
                -candidate["total_points_final_phase"],
            ),
        )
        results.append(
            {
                "budget_points": budget,
                "feasible": True,
                **best,
            }
        )

    output = {
        "inputs": {
            "tum": str(TUM_INPUT),
            "gold_coast": str(GC_INPUT),
        },
        "tum_final_scene_scale": TUM_FINAL_SCENE_SCALE,
        "budgets_points": list(BUDGETS),
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")

    print(
        "| Budget | TUM voxel | GC voxel | Geometry | Semantics | Total | "
        "Final-phase points |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        if not result["feasible"]:
            print(f"| {result['budget_points'] / 1e6:.0f}M | infeasible | - | - | - | - | - |")
            continue
        print(
            f"| {result['budget_points'] / 1e6:.0f}M | "
            f"{result['tum_voxel']:.3f} | {result['gc_voxel']:.3f} | "
            f"{result['predicted_geometry']:.4f} | "
            f"{result['predicted_semantics']:.4f} | "
            f"{result['predicted_total']:.4f} | "
            f"{result['total_points_final_phase'] / 1e6:.2f}M |"
        )


if __name__ == "__main__":
    main()
