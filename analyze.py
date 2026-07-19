#!/usr/bin/env python3
"""Reproducible EDMD validation on exact public Pythia-14m checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "EleutherAI/pythia-14m"
CHECKPOINTS = (
    {
        "label": "step1000",
        "step": 1_000,
        "commit": "5b020995bfc7aee2931b0f35bd70cf7ee8b1db62",
    },
    {
        "label": "step10000",
        "step": 10_000,
        "commit": "b9935f34c34c4bddaa99bed4c2ed3fc8e67c7504",
    },
    {
        "label": "step143000",
        "step": 143_000,
        "commit": "f1545025bb394553a7f4e547db0874886f05ef9c",
    },
)
SETTINGS = {
    "seed": 20_260_718,
    "max_tokens": 128,
    "burn_in_tokens": 4,
    "pca_rank": 16,
    "ridge": 0.001,
    "shuffle_repeats": 50,
    "horizons": [1, 2, 4, 8],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for generated results (default: ./results).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only model files already present in the Hugging Face cache.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["all", *(item["label"] for item in CHECKPOINTS)],
        default="all",
        help="Run every predeclared checkpoint or one named checkpoint.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    prompts = payload["prompts"]
    ids = [item["id"] for item in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt identifiers must be unique.")
    if {item["split"] for item in prompts} != {"train", "test"}:
        raise ValueError("The corpus must contain explicit train and test prompts.")
    return prompts


def set_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def extract_trajectories(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    num_layers = int(model.config.num_hidden_layers)
    block_index = (num_layers - 1) // 2
    hidden_state_index = block_index + 1
    trajectories: list[dict[str, Any]] = []

    for prompt in prompts:
        encoded = tokenizer(
            prompt["text"],
            return_tensors="pt",
            truncation=True,
            max_length=SETTINGS["max_tokens"],
            add_special_tokens=False,
        )
        token_count = int(encoded["input_ids"].shape[1])
        if token_count <= SETTINGS["burn_in_tokens"] + max(SETTINGS["horizons"]):
            raise ValueError(f"Prompt {prompt['id']} has only {token_count} tokens.")
        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        states = (
            output.hidden_states[hidden_state_index][0]
            .detach()
            .to(dtype=torch.float32, device="cpu")
            .numpy()
        )
        trajectories.append(
            {
                **prompt,
                "token_count": token_count,
                "states": states[SETTINGS["burn_in_tokens"] :],
            }
        )

    layer_info = {
        "num_transformer_blocks": num_layers,
        "selected_block_zero_based": block_index,
        "hidden_state_tuple_index": hidden_state_index,
    }
    return trajectories, layer_info


def snapshot_pairs(
    trajectories: list[dict[str, Any]], split: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = [item["states"] for item in trajectories if item["split"] == split]
    return (
        np.concatenate([states[:-1] for states in selected], axis=0),
        np.concatenate([states[1:] for states in selected], axis=0),
    )


def fit_pca(x: np.ndarray, y: np.ndarray, rank: int) -> dict[str, np.ndarray | float]:
    pool = np.concatenate([x, y], axis=0).astype(np.float64)
    mean = pool.mean(axis=0)
    centered = pool - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    actual_rank = min(rank, vt.shape[0])
    components = vt[:actual_rank]
    projected = centered @ components.T
    scale = projected.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    total_energy = float(np.square(singular_values).sum())
    retained = float(np.square(singular_values[:actual_rank]).sum() / total_energy)
    return {
        "mean": mean,
        "components": components,
        "scale": scale,
        "variance_retained": retained,
        "singular_values": singular_values,
    }


def transform_pca(states: np.ndarray, pca: dict[str, Any]) -> np.ndarray:
    return ((states.astype(np.float64) - pca["mean"]) @ pca["components"].T) / pca[
        "scale"
    ]


def raw_dictionary(z: np.ndarray) -> np.ndarray:
    return np.concatenate([z, np.square(z), np.tanh(z)], axis=1)


def fit_dictionary_normalizer(z_x: np.ndarray, z_y: np.ndarray) -> dict[str, np.ndarray]:
    pool = np.concatenate([raw_dictionary(z_x), raw_dictionary(z_y)], axis=0)
    mean = pool.mean(axis=0)
    scale = pool.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return {"mean": mean, "scale": scale}


def transform_dictionary(z: np.ndarray, normalizer: dict[str, np.ndarray]) -> np.ndarray:
    return (raw_dictionary(z) - normalizer["mean"]) / normalizer["scale"]


def decode_linear_observables(
    phi: np.ndarray, normalizer: dict[str, np.ndarray], rank: int
) -> np.ndarray:
    raw_linear = phi[:, :rank] * normalizer["scale"][:rank]
    return raw_linear + normalizer["mean"][:rank]


def fit_ridge_operator(phi_x: np.ndarray, phi_y: np.ndarray, ridge: float) -> np.ndarray:
    gram = phi_x.T @ phi_x
    penalty = ridge * max(1.0, float(np.trace(gram) / gram.shape[0]))
    return np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0], dtype=np.float64),
        phi_x.T @ phi_y,
    )


def prediction_metrics(
    predicted: np.ndarray, actual: np.ndarray, baseline: np.ndarray
) -> dict[str, float]:
    mse = float(np.mean(np.square(predicted - actual)))
    baseline_mse = float(np.mean(np.square(baseline - actual)))
    nrmse = math.sqrt(mse / baseline_mse) if baseline_mse > 0 else float("nan")
    return {
        "rmse": math.sqrt(mse),
        "normalized_rmse": nrmse,
        "r_squared_vs_mean": 1.0 - (mse / baseline_mse) if baseline_mse > 0 else float("nan"),
    }


def evaluate_horizons(
    trajectories: list[dict[str, Any]],
    pca: dict[str, Any],
    normalizer: dict[str, np.ndarray],
    operator: np.ndarray,
    train_target_mean: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    rank = int(pca["components"].shape[0])
    test_states = [item["states"] for item in trajectories if item["split"] == "test"]
    for horizon in SETTINGS["horizons"]:
        predicted_parts = []
        actual_parts = []
        operator_h = np.linalg.matrix_power(operator, horizon)
        for states in test_states:
            z = transform_pca(states, pca)
            start_phi = transform_dictionary(z[:-horizon], normalizer)
            predicted_parts.append(
                decode_linear_observables(start_phi @ operator_h, normalizer, rank)
            )
            actual_parts.append(z[horizon:])
        predicted = np.concatenate(predicted_parts, axis=0)
        actual = np.concatenate(actual_parts, axis=0)
        baseline = np.broadcast_to(train_target_mean, actual.shape)
        metrics[str(horizon)] = prediction_metrics(predicted, actual, baseline)
    return metrics


def evaluate_categories(
    trajectories: list[dict[str, Any]],
    pca: dict[str, Any],
    normalizer: dict[str, np.ndarray],
    operator: np.ndarray,
    train_target_mean: np.ndarray,
) -> dict[str, dict[str, Any]]:
    rank = int(pca["components"].shape[0])
    categories = sorted(
        {item["category"] for item in trajectories if item["split"] == "test"}
    )
    results: dict[str, dict[str, Any]] = {}
    for category in categories:
        members = [
            item
            for item in trajectories
            if item["split"] == "test" and item["category"] == category
        ]
        z_parts = [transform_pca(item["states"], pca) for item in members]
        z_x = np.concatenate([z[:-1] for z in z_parts], axis=0)
        z_y = np.concatenate([z[1:] for z in z_parts], axis=0)
        phi_x = transform_dictionary(z_x, normalizer)
        prediction = decode_linear_observables(phi_x @ operator, normalizer, rank)
        baseline = np.broadcast_to(train_target_mean, z_y.shape)
        results[category] = {
            "prompt_count": len(members),
            "snapshot_pairs": int(z_y.shape[0]),
            "edmd": prediction_metrics(prediction, z_y, baseline),
            "persistence": prediction_metrics(z_x, z_y, baseline),
        }
    return results


def persistence_jackknife(
    trajectories: list[dict[str, Any]],
    pca: dict[str, Any],
    normalizer: dict[str, np.ndarray],
    operator: np.ndarray,
    train_target_mean: np.ndarray,
) -> dict[str, Any]:
    rank = int(pca["components"].shape[0])
    rows = []
    for item in [t for t in trajectories if t["split"] == "test"]:
        z = transform_pca(item["states"], pca)
        phi_x = transform_dictionary(z[:-1], normalizer)
        predicted = decode_linear_observables(phi_x @ operator, normalizer, rank)
        rows.append(
            {
                "id": item["id"],
                "edmd_sse": float(np.square(predicted - z[1:]).sum()),
                "persistence_sse": float(np.square(z[:-1] - z[1:]).sum()),
                "baseline_sse": float(np.square(train_target_mean - z[1:]).sum()),
            }
        )

    def nrmse_pair(selected: list[dict[str, Any]]) -> tuple[float, float]:
        base = sum(row["baseline_sse"] for row in selected)
        return (
            math.sqrt(sum(row["edmd_sse"] for row in selected) / base),
            math.sqrt(sum(row["persistence_sse"] for row in selected) / base),
        )

    full_edmd, full_persistence = nrmse_pair(rows)
    full_beats = bool(full_edmd < full_persistence)
    drops = []
    for index, row in enumerate(rows):
        edmd, persistence = nrmse_pair(rows[:index] + rows[index + 1 :])
        drops.append(
            {
                "dropped_prompt": row["id"],
                "edmd_normalized_rmse": edmd,
                "persistence_normalized_rmse": persistence,
                "edmd_beats_persistence": bool(edmd < persistence),
            }
        )
    return {
        "full_edmd_beats_persistence": full_beats,
        "flip_count": sum(
            drop["edmd_beats_persistence"] != full_beats for drop in drops
        ),
        "prompt_count": len(rows),
        "drops": drops,
    }


def spectral_metrics(operator: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    eigenvalues = np.linalg.eigvals(operator)
    magnitudes = np.abs(eigenvalues)
    stable = (magnitudes > 1e-12) & (magnitudes < 1.0)
    half_lives = np.log(0.5) / np.log(magnitudes[stable])
    persistent = (magnitudes >= 0.95) & (magnitudes < 1.0)
    metrics = {
        "spectral_radius": float(magnitudes.max()),
        "mean_eigenvalue_magnitude": float(magnitudes.mean()),
        "stable_fraction": float(np.mean(magnitudes < 1.0)),
        "persistent_stable_mode_count": int(persistent.sum()),
        "median_stable_half_life_tokens": (
            float(np.median(half_lives)) if half_lives.size else None
        ),
        "max_stable_half_life_tokens": (
            float(np.max(half_lives)) if half_lives.size else None
        ),
    }
    return eigenvalues, metrics


def run_checkpoint(
    checkpoint: dict[str, Any],
    tokenizer: Any,
    prompts: list[dict[str, str]],
    offline: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    print(f"\nLoading {checkpoint['label']} ({checkpoint['commit']})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=checkpoint["commit"],
        local_files_only=offline,
    )
    model.eval()
    model.to("cpu")

    trajectories, layer_info = extract_trajectories(model, tokenizer, prompts)
    train_x, train_y = snapshot_pairs(trajectories, "train")
    test_x, test_y = snapshot_pairs(trajectories, "test")
    pca = fit_pca(train_x, train_y, SETTINGS["pca_rank"])
    z_train_x = transform_pca(train_x, pca)
    z_train_y = transform_pca(train_y, pca)
    z_test_x = transform_pca(test_x, pca)
    z_test_y = transform_pca(test_y, pca)
    normalizer = fit_dictionary_normalizer(z_train_x, z_train_y)
    phi_train_x = transform_dictionary(z_train_x, normalizer)
    phi_train_y = transform_dictionary(z_train_y, normalizer)
    phi_test_x = transform_dictionary(z_test_x, normalizer)
    operator = fit_ridge_operator(phi_train_x, phi_train_y, SETTINGS["ridge"])

    rank = int(pca["components"].shape[0])
    train_target_mean = z_train_y.mean(axis=0)
    test_prediction = decode_linear_observables(
        phi_test_x @ operator, normalizer, rank
    )
    test_baseline = np.broadcast_to(train_target_mean, z_test_y.shape)
    heldout = prediction_metrics(test_prediction, z_test_y, test_baseline)
    persistence = prediction_metrics(z_test_x, z_test_y, test_baseline)
    persistence_robustness_pass = bool(
        heldout["normalized_rmse"] < persistence["normalized_rmse"]
    )

    rng = np.random.default_rng(SETTINGS["seed"] + int(checkpoint["step"]))
    shuffled_errors = []
    for _ in range(SETTINGS["shuffle_repeats"]):
        shuffled_y = phi_train_y[rng.permutation(phi_train_y.shape[0])]
        shuffled_operator = fit_ridge_operator(
            phi_train_x, shuffled_y, SETTINGS["ridge"]
        )
        shuffled_prediction = decode_linear_observables(
            phi_test_x @ shuffled_operator, normalizer, rank
        )
        shuffled_errors.append(
            prediction_metrics(shuffled_prediction, z_test_y, test_baseline)[
                "normalized_rmse"
            ]
        )
    shuffled_array = np.asarray(shuffled_errors, dtype=np.float64)
    empirical_p = float(
        (1 + np.sum(shuffled_array <= heldout["normalized_rmse"]))
        / (1 + shuffled_array.size)
    )
    primary_pass = bool(heldout["normalized_rmse"] < 1.0 and empirical_p <= 0.05)

    horizons = evaluate_horizons(
        trajectories, pca, normalizer, operator, train_target_mean
    )
    category_metrics = evaluate_categories(
        trajectories, pca, normalizer, operator, train_target_mean
    )
    jackknife = persistence_jackknife(
        trajectories, pca, normalizer, operator, train_target_mean
    )
    eigenvalues, spectrum = spectral_metrics(operator)
    prompt_counts = {
        split: sum(item["split"] == split for item in trajectories)
        for split in ("train", "test")
    }
    category_counts = {
        category: sum(item["category"] == category for item in trajectories)
        for category in sorted({item["category"] for item in trajectories})
    }
    token_counts = {
        "minimum": min(item["token_count"] for item in trajectories),
        "maximum": max(item["token_count"] for item in trajectories),
        "total": sum(item["token_count"] for item in trajectories),
    }
    result = {
        **checkpoint,
        "layer": layer_info,
        "hidden_size": int(model.config.hidden_size),
        "prompt_counts": prompt_counts,
        "category_counts": category_counts,
        "token_counts": token_counts,
        "train_snapshot_pairs": int(train_x.shape[0]),
        "test_snapshot_pairs": int(test_x.shape[0]),
        "pca_variance_retained": float(pca["variance_retained"]),
        "heldout_one_step": heldout,
        "mean_baseline_normalized_rmse": 1.0,
        "persistence_baseline": persistence,
        "post_run_persistence_robustness_pass": persistence_robustness_pass,
        "shuffled_control": {
            "repetitions": int(shuffled_array.size),
            "normalized_rmse_mean": float(shuffled_array.mean()),
            "normalized_rmse_std": float(shuffled_array.std(ddof=1)),
            "normalized_rmse_min": float(shuffled_array.min()),
            "normalized_rmse_max": float(shuffled_array.max()),
            "normalized_rmse_values": shuffled_array.tolist(),
            "one_sided_empirical_p": empirical_p,
        },
        "primary_gate_pass": primary_pass,
        "multi_step": horizons,
        "heldout_by_category": category_metrics,
        "post_run_persistence_jackknife": jackknife,
        "spectrum": spectrum,
        "eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in eigenvalues
        ],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    print(
        f"{checkpoint['label']}: held-out nRMSE={heldout['normalized_rmse']:.4f}, "
        f"persistence={persistence['normalized_rmse']:.4f}, "
        f"shuffle={shuffled_array.mean():.4f} +/- {shuffled_array.std(ddof=1):.4f}, "
        f"p={empirical_p:.4f}, pass={primary_pass}",
        flush=True,
    )
    del model
    return result


def write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "checkpoint",
        "step",
        "commit",
        "selected_block_zero_based",
        "train_pairs",
        "test_pairs",
        "pca_variance_retained",
        "heldout_nrmse",
        "heldout_r_squared_vs_mean",
        "persistence_nrmse",
        "beats_persistence_post_run",
        "shuffle_nrmse_mean",
        "shuffle_nrmse_std",
        "empirical_p",
        "primary_gate_pass",
        "spectral_radius",
        "stable_fraction",
        "persistent_stable_mode_count",
        "median_stable_half_life_tokens",
        "max_stable_half_life_tokens",
        "horizon_2_nrmse",
        "horizon_4_nrmse",
        "horizon_8_nrmse",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            spectrum = result["spectrum"]
            writer.writerow(
                {
                    "checkpoint": result["label"],
                    "step": result["step"],
                    "commit": result["commit"],
                    "selected_block_zero_based": result["layer"][
                        "selected_block_zero_based"
                    ],
                    "train_pairs": result["train_snapshot_pairs"],
                    "test_pairs": result["test_snapshot_pairs"],
                    "pca_variance_retained": result["pca_variance_retained"],
                    "heldout_nrmse": result["heldout_one_step"]["normalized_rmse"],
                    "heldout_r_squared_vs_mean": result["heldout_one_step"][
                        "r_squared_vs_mean"
                    ],
                    "persistence_nrmse": result["persistence_baseline"][
                        "normalized_rmse"
                    ],
                    "beats_persistence_post_run": result[
                        "post_run_persistence_robustness_pass"
                    ],
                    "shuffle_nrmse_mean": result["shuffled_control"][
                        "normalized_rmse_mean"
                    ],
                    "shuffle_nrmse_std": result["shuffled_control"][
                        "normalized_rmse_std"
                    ],
                    "empirical_p": result["shuffled_control"][
                        "one_sided_empirical_p"
                    ],
                    "primary_gate_pass": result["primary_gate_pass"],
                    "spectral_radius": spectrum["spectral_radius"],
                    "stable_fraction": spectrum["stable_fraction"],
                    "persistent_stable_mode_count": spectrum[
                        "persistent_stable_mode_count"
                    ],
                    "median_stable_half_life_tokens": spectrum[
                        "median_stable_half_life_tokens"
                    ],
                    "max_stable_half_life_tokens": spectrum[
                        "max_stable_half_life_tokens"
                    ],
                    "horizon_2_nrmse": result["multi_step"]["2"][
                        "normalized_rmse"
                    ],
                    "horizon_4_nrmse": result["multi_step"]["4"][
                        "normalized_rmse"
                    ],
                    "horizon_8_nrmse": result["multi_step"]["8"][
                        "normalized_rmse"
                    ],
                }
            )


def write_interpretation(results: list[dict[str, Any]], path: Path) -> None:
    pass_count = sum(result["primary_gate_pass"] for result in results)
    persistence_pass_count = sum(
        result["post_run_persistence_robustness_pass"] for result in results
    )
    lines = [
        "# Result and bounded interpretation",
        "",
        (
            f"The originally pre-specified primary gate passed at **{pass_count} of "
            f"{len(results)}** analyzed public checkpoints."
        ),
        "",
        "| Checkpoint | EDMD nRMSE | Persistence nRMSE | Shuffled nRMSE (mean +/- SD) | p | Original gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        heldout = result["heldout_one_step"]["normalized_rmse"]
        control = result["shuffled_control"]
        lines.append(
            f"| {result['label']} | {heldout:.4f} | "
            f"{result['persistence_baseline']['normalized_rmse']:.4f} | "
            f"{control['normalized_rmse_mean']:.4f} +/- "
            f"{control['normalized_rmse_std']:.4f} | "
            f"{control['one_sided_empirical_p']:.4f} | "
            f"{'pass' if result['primary_gate_pass'] else 'fail'} |"
        )
    repeats = int(results[0]["shuffled_control"]["repetitions"])
    shuffle_beat_counts = {
        result["label"]: sum(
            value <= result["heldout_one_step"]["normalized_rmse"]
            for value in result["shuffled_control"]["normalized_rmse_values"]
        )
        for result in results
    }
    lines.extend(
        [
            "",
            "Normalized RMSE below 1.0 beats the training-target mean predictor. "
            "The one-sided empirical p-value asks how often a shuffled-target "
            "fit achieved an error at least as low as the real-pair fit.",
            "",
        ]
    )
    if all(count == 0 for count in shuffle_beat_counts.values()):
        lines.append(
            f"At every checkpoint, 0 of {repeats} shuffled-target fits matched "
            f"the real fit; the add-one correction gives p = 1/{repeats + 1} "
            f"~= {1 / (repeats + 1):.4f}, the smallest value this test can "
            f"report."
        )
    else:
        lines.append(
            "Shuffled fits matching or beating the real fit: "
            + "; ".join(
                f"{label}: {count} of {repeats}"
                for label, count in shuffle_beat_counts.items()
            )
            + "."
        )
    category_names = ("periodic", "progressive", "prose")
    category_wins = {
        name: sum(
            result["heldout_by_category"][name]["edmd"]["normalized_rmse"]
            < result["heldout_by_category"][name]["persistence"]["normalized_rmse"]
            for result in results
        )
        for name in category_names
    }
    lines.extend(
        [
            "",
            "## Post-run persistence robustness (amendments v1.1 and v1.2)",
            "",
            (
                f"Across all held-out prompts, EDMD outperformed the "
                f"identity/persistence predictor at **{persistence_pass_count} of "
                f"{len(results)}** checkpoints. This comparison was specified "
                f"only after the original primary result was viewed. The "
                f"advantage is category-dependent, as the breakdown below shows."
            ),
            "",
            "### Held-out category breakdown (descriptive)",
            "",
            (
                "| Checkpoint | Periodic EDMD | Periodic persist. | "
                "Progressive EDMD | Progressive persist. | Prose EDMD | "
                "Prose persist. |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        categories = result["heldout_by_category"]
        cells = []
        for name in category_names:
            cells.append(f"{categories[name]['edmd']['normalized_rmse']:.4f}")
            cells.append(
                f"{categories[name]['persistence']['normalized_rmse']:.4f}"
            )
        lines.append(f"| {result['label']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                "Checkpoints where EDMD beat persistence within each category -- "
                + "; ".join(
                    f"{name}: {category_wins[name]} of {len(results)}"
                    for name in category_names
                )
                + "."
            ),
        ]
    )
    weak_categories = [
        f"{result['label']} {name} "
        f"({result['heldout_by_category'][name]['edmd']['normalized_rmse']:.4f})"
        for result in results
        for name in category_names
        if result["heldout_by_category"][name]["edmd"]["normalized_rmse"] >= 1.0
    ]
    lines.extend(
        [
            "",
            (
                "Held-out categories with EDMD error at or above the "
                "mean-prediction baseline: "
                + (", ".join(weak_categories) if weak_categories else "none")
                + "."
            ),
            "",
            "### Leave-one-prompt-out sensitivity (v1.2)",
            "",
        ]
    )
    for result in results:
        jackknife = result["post_run_persistence_jackknife"]
        flipped = [
            drop["dropped_prompt"]
            for drop in jackknife["drops"]
            if drop["edmd_beats_persistence"]
            != jackknife["full_edmd_beats_persistence"]
        ]
        if flipped:
            lines.append(
                f"- {result['label']}: the aggregate EDMD-versus-persistence "
                f"comparison reverses when any one of {len(flipped)} of "
                f"{jackknife['prompt_count']} held-out prompts is removed "
                f"({', '.join(flipped)})."
            )
        else:
            lines.append(
                f"- {result['label']}: the aggregate comparison is unchanged "
                f"under all {jackknife['prompt_count']} single-prompt removals."
            )
    retained = ", ".join(
        f"{100 * result['pca_variance_retained']:.1f}%" for result in results
    )
    lines.extend(
        [
            "",
            "## Spectral comparability caveat",
            "",
            (
                f"Each checkpoint's rank-{SETTINGS['pca_rank']} PCA basis is fit "
                f"independently, and the retained training-state variance "
                f"differs across checkpoints ({retained}). Cross-checkpoint "
                f"spectral comparisons therefore conflate changes in the "
                f"underlying dynamics with changes in the captured subspace, "
                f"and they remain descriptive."
            ),
        ]
    )
    lines.append("")
    if pass_count == len(results):
        lines.append(
            "Under the fixed protocol, real temporal pairing carried "
            "out-of-sample predictive structure at every analyzed checkpoint."
        )
    elif pass_count:
        lines.append(
            "Under the fixed protocol, real temporal pairing carried "
            "out-of-sample predictive structure at some, but not all, analyzed "
            "checkpoints."
        )
    else:
        lines.append(
            "Under the fixed protocol, the experiment did not establish "
            "out-of-sample temporal structure beyond both controls."
        )
    lines.extend(
        [
            "",
            "## Defensible conclusion",
            "",
            (
                f"A fixed, disclosed EDMD pipeline recovered held-out one-step "
                f"temporal structure from the hidden activations of "
                f"{pass_count} of {len(results)} public Pythia-14m training "
                f"checkpoints, outperforming a train-mean predictor and "
                f"{repeats} shuffled-pairing controls at each passing "
                f"checkpoint. On held-out prose it also outperformed state "
                f"persistence at {category_wins['prose']} of {len(results)} "
                f"checkpoints; the category tallies above show where "
                f"persistence was better. Cross-checkpoint spectral "
                f"differences remain descriptive."
            ),
            "",
            (
                "The predictive-control result above is the evidentiary gate, "
                "not the visual appearance of the spectrum."
            ),
            "",
            "## What this does not prove",
            "",
            (
                "This result does not show that the public transformer implements "
                "a linear recurrence, that an eigenmode has a particular semantic "
                "meaning, or that a spectral statistic alone identifies a causal "
                "memory bottleneck. It does not validate claims about another model."
            ),
            "",
            (
                "A bottleneck claim would need a separate behavioral or causal "
                "test: predict a specific retention failure from the spectrum, "
                "intervene on the implicated mechanism, and show that the predicted "
                "failure changes while suitable controls do not."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [item["label"] for item in results]
    x = np.arange(len(labels))
    heldout = np.array(
        [item["heldout_one_step"]["normalized_rmse"] for item in results]
    )
    control_mean = np.array(
        [item["shuffled_control"]["normalized_rmse_mean"] for item in results]
    )
    control_std = np.array(
        [item["shuffled_control"]["normalized_rmse_std"] for item in results]
    )
    persistence = np.array(
        [item["persistence_baseline"]["normalized_rmse"] for item in results]
    )

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, max(3, len(results)))
    ax = fig.add_subplot(grid[0, :])
    ax.plot(x, heldout, marker="o", linewidth=2.2, label="Real temporal pairs")
    ax.plot(
        x,
        persistence,
        marker="^",
        linewidth=1.7,
        label="Identity / persistence predictor",
    )
    ax.errorbar(
        x,
        control_mean,
        yerr=control_std,
        marker="s",
        capsize=4,
        linewidth=1.7,
        label="Shuffled targets (mean +/- SD)",
    )
    ax.axhline(1.0, color="0.35", linestyle="--", label="Mean-prediction baseline")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out one-step normalized RMSE (lower is better)")
    ax.set_title("Predictive validation is the evidentiary gate")
    ax.grid(alpha=0.25)
    ax.legend()

    unit_circle = np.linspace(0, 2 * np.pi, 500)
    for index, result in enumerate(results):
        spectrum_ax = fig.add_subplot(grid[1, index])
        eigenvalues = np.array(
            [complex(item["real"], item["imag"]) for item in result["eigenvalues"]]
        )
        spectrum_ax.plot(
            np.cos(unit_circle), np.sin(unit_circle), color="0.5", linestyle="--"
        )
        spectrum_ax.scatter(
            eigenvalues.real,
            eigenvalues.imag,
            c=np.abs(eigenvalues),
            cmap="viridis",
            s=24,
            alpha=0.82,
        )
        limit = max(1.1, float(np.abs(eigenvalues).max()) * 1.08)
        spectrum_ax.set_xlim(-limit, limit)
        spectrum_ax.set_ylim(-limit, limit)
        spectrum_ax.set_aspect("equal", adjustable="box")
        spectrum_ax.axhline(0, color="0.8", linewidth=0.8)
        spectrum_ax.axvline(0, color="0.8", linewidth=0.8)
        spectrum_ax.set_title(result["label"])
        spectrum_ax.set_xlabel("Real(lambda)")
        if index == 0:
            spectrum_ax.set_ylabel("Imag(lambda)")
        spectrum_ax.grid(alpha=0.2)
    fig.suptitle("Public Pythia-14m EDMD validation and fitted spectra", fontsize=15)
    fig.savefig(output_dir / "edmd_validation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    retained = [100 * result["pca_variance_retained"] for result in results]
    bars = ax.bar(labels, retained, color="#4263a8")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Training-state variance retained (%)")
    ax.set_title("Fixed 16-dimensional PCA diagnostic")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "pca_variance.png", dpi=180)
    plt.close(fig)


def package_versions() -> dict[str, str]:
    names = (
        "torch",
        "numpy",
        "matplotlib",
        "transformers",
        "huggingface-hub",
        "safetensors",
    )
    return {name: importlib.metadata.version(name) for name in names}


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    protocol_path = root / "PROTOCOL.md"
    prompts_path = root / "prompts.json"
    script_path = Path(__file__).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(SETTINGS["seed"])
    prompts = load_corpus(prompts_path)

    selected = list(CHECKPOINTS)
    if args.checkpoint != "all":
        selected = [item for item in selected if item["label"] == args.checkpoint]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=CHECKPOINTS[-1]["commit"],
        local_files_only=args.offline,
    )
    results = [
        run_checkpoint(item, tokenizer, prompts, args.offline) for item in selected
    ]

    summary = {
        "title": "Public Checkpoint EDMD Demonstration",
        "model_id": MODEL_ID,
        "settings": SETTINGS,
        "primary_gate_pass_count": sum(
            item["primary_gate_pass"] for item in results
        ),
        "checkpoint_count": len(results),
        "results": results,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    write_summary_csv(results, args.output_dir / "checkpoint_summary.csv")
    write_interpretation(results, args.output_dir / "interpretation.md")
    plot_results(results, args.output_dir)

    manifest = {
        "model_id": MODEL_ID,
        "checkpoints": selected,
        "settings": SETTINGS,
        "offline_mode": args.offline,
        "file_sha256": {
            "PROTOCOL.md": sha256_file(protocol_path),
            "prompts.json": sha256_file(prompts_path),
            "analyze.py": sha256_file(script_path),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    with (args.output_dir / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"\nResults written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
