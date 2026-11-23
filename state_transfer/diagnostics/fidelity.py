"""Evaluate saved held-out spans against actual transmitted packets on CPU."""

import hashlib

import numpy as np

import reference_decoder as numeric
from evidence import completed_manifest, read, rows, sha, require, write_csv


def compute(plan_file, run, output):
    plan = read(plan_file)
    manifest = completed_manifest(run, "complete_isolated_heldout_study")

    def sealed(path):
        name = path.resolve().relative_to(run.resolve()).as_posix()
        require(
            sha(path) == manifest[name]["sha256"],
            "Study evidence digest differs: " + name,
        )

    sealed(run / "input_receipt.json")
    require(
        read(run / "input_receipt.json")["plan_sha256"] == sha(plan_file),
        "Study output belongs to a different protocol",
    )
    all_rows, summary = [], []
    for model, settings in plan["models"].items():
        sender = run / (model + "_sender") / "runtime/stage"
        receiver = run / (model + "_pca_receiver") / "runtime/stage"
        sealed(sender / "encodings.jsonl")
        sealed(receiver / "rows.jsonl")
        captures = rows(sender / "encodings.jsonl")
        recipient = {
            x["family_id"]: x
            for x in rows(receiver / "rows.jsonl")
            if x["condition"] == "pca_generated"
        }
        for task, binding in settings["tasks"].items():
            basis_path = plan_file.parent / "assets" / binding["basis_file"]
            require(
                sha(basis_path) == binding["basis_sha256"],
                "Frozen basis digest differs",
            )
            with np.load(basis_path, allow_pickle=False) as basis:
                mean = basis["mean"].astype(np.float64)
                components = basis["components"].astype(np.float64)
                require(
                    components.shape == (512, len(mean))
                    and int(basis["layer"]) == settings["layer"],
                    "Basis shape or target layer differs",
                )
            selected = [
                x
                for x in captures
                if x["task"] == task
                and x["kind"] == "generated"
                and x["variant"] == "original"
            ]
            require(
                len(selected) == len({x["family_id"] for x in selected}) == 64,
                "Expected 64 distinct original families for this task",
            )
            cell = []
            for capture in selected:
                source_path = sender / capture["array_file"]
                sealed(source_path)
                require(
                    sha(source_path) == capture["array_file_sha256"],
                    "Sender array file digest differs",
                )
                source = np.load(source_path, allow_pickle=False)
                require(
                    source.dtype == np.float32
                    and list(source.shape) == capture["encoding"]["shape"],
                    "Unexpected sender tensor shape or dtype",
                )
                require(
                    hashlib.sha256(source.astype("<f4").tobytes()).hexdigest()
                    == capture["encoding"]["source_array_sha256"],
                    "Sender tensor data digest differs",
                )
                require(np.isfinite(source).all(), "Nonfinite sender tensor")
                packet_info = capture["packets"]["pca"]
                packet_path = sender / packet_info["file"]
                sealed(packet_path)
                require(
                    sha(packet_path) == packet_info["sha256"],
                    "Sender packet digest differs",
                )
                packet = packet_path.read_bytes()
                require(
                    len(packet) * 8 == packet_info["wire_bits"],
                    "Reported packet bit count differs",
                )
                require(
                    packet[:4] == b"PCA1"
                    and packet[4:36].hex() == binding["basis_sha256"],
                    "PCA packet missing or bound to the wrong basis",
                )
                coeff = numeric.decode_packet(
                    packet[36:],
                    settings["pca_codec"],
                    settings["layer"],
                    0,
                    (len(source), 512),
                )
                reconstructed64 = coeff.astype(np.float64) @ components + mean
                alternate64 = (
                    np.einsum(
                        "ij,jk->ik",
                        coeff.astype(np.float64),
                        components,
                        optimize=False,
                    )
                    + mean
                )
                max_summation_difference = float(
                    np.max(np.abs(reconstructed64 - alternate64))
                )
                require(
                    max_summation_difference < 1e-10,
                    "Matrix multiply and independent summation disagree",
                )
                reconstructed = reconstructed64.astype(np.float32)
                receiver_result = recipient[capture["family_id"]]["result"]
                require(
                    receiver_result["packet_sha256"] == packet_info["sha256"],
                    "Receiver packet differs from the sender packet",
                )
                require(
                    numeric.digest(numeric.f32bytes(coeff))
                    == receiver_result["decoded_coefficients_sha256"],
                    "Decoded coefficient digest differs from receiver record",
                )
                recorded_hash_matches = (
                    numeric.digest(numeric.f32bytes(reconstructed))
                    == receiver_result["reconstructed_hidden_sha256"]
                )
                alternate_hash_matches = (
                    numeric.digest(numeric.f32bytes(alternate64.astype(np.float32)))
                    == receiver_result["reconstructed_hidden_sha256"]
                )
                require(
                    recorded_hash_matches or alternate_hash_matches,
                    "Neither reconstruction method matches receiver state",
                )
                centered = source.astype(np.float64) - mean
                exact_projection = centered @ components.T @ components + mean
                squared_centered = float(np.sum(centered**2))
                squared_source = float(np.sum(source.astype(np.float64) ** 2))
                squared_error_before = float(np.sum((exact_projection - source) ** 2))
                squared_error_after = float(
                    np.sum((reconstructed.astype(np.float64) - source) ** 2)
                )
                alternate_error = float(
                    np.sum(
                        (alternate64.astype(np.float32).astype(np.float64) - source)
                        ** 2
                    )
                )
                require(
                    abs(squared_error_after - alternate_error) / squared_centered
                    < 1e-10,
                    "Relative errors depend on summation method",
                )
                require(
                    0 < squared_centered
                    and 0 <= squared_error_before <= squared_error_after + 1e-10,
                    "Squared errors violate the projection bound",
                )
                result = dict(
                    model=model,
                    task=task,
                    family_id=capture["family_id"],
                    token_rows=len(source),
                    metric_method="numpy_float64_matmul_then_float32_all_families",
                    receiver_reconstruction_hash_matches=recorded_hash_matches,
                    alternate_summation_receiver_hash_matches=alternate_hash_matches,
                    independent_summation_max_abs_difference=max_summation_difference,
                    squared_source_centered_about_frozen_calibration_mean=squared_centered,
                    squared_source_uncentered=squared_source,
                    squared_error_projection=squared_error_before,
                    squared_error_after_packet_quantization=squared_error_after,
                    centered_rse_projection=squared_error_before / squared_centered,
                    centered_rse_after_packet_quantization=squared_error_after
                    / squared_centered,
                    independent_einsum_centered_rse_after_packet_quantization=alternate_error
                    / squared_centered,
                    uncentered_rse_after_packet_quantization=squared_error_after
                    / squared_source,
                )
                cell.append(result)
            all_rows.extend(cell)
            energy = sum(
                x["squared_source_centered_about_frozen_calibration_mean"] for x in cell
            )
            before = sum(x["squared_error_projection"] for x in cell) / energy
            after = (
                sum(x["squared_error_after_packet_quantization"] for x in cell) / energy
            )
            after_per_family = [
                x["centered_rse_after_packet_quantization"] for x in cell
            ]
            summary.append(
                dict(
                    model=model,
                    task=task,
                    n_families=64,
                    n_source_arrays=64,
                    token_rows=sum(x["token_rows"] for x in cell),
                    rank=512,
                    codec=settings["pca_codec"],
                    receiver_reconstruction_hash_matches=sum(
                        x["receiver_reconstruction_hash_matches"] for x in cell
                    ),
                    either_summation_receiver_hash_matches=sum(
                        x["receiver_reconstruction_hash_matches"]
                        or x["alternate_summation_receiver_hash_matches"]
                        for x in cell
                    ),
                    independent_summation_max_abs_difference=max(
                        x["independent_summation_max_abs_difference"] for x in cell
                    ),
                    pooled_centered_rse_projection=before,
                    pooled_centered_rse_projection_percent=100 * before,
                    pooled_centered_rse_after_packet_quantization=after,
                    pooled_centered_rse_after_packet_quantization_percent=100 * after,
                    mean_family_centered_rse_after_packet_quantization=float(
                        np.mean(after_per_family)
                    ),
                    median_family_centered_rse_after_packet_quantization=float(
                        np.median(after_per_family)
                    ),
                    min_family_centered_rse_after_packet_quantization=min(
                        after_per_family
                    ),
                    max_family_centered_rse_after_packet_quantization=max(
                        after_per_family
                    ),
                )
            )
            print(f"Reconstructed {len(cell)} spans ({model}, {task})", flush=True)
    require(len(all_rows) == 256, "Expected 256 held-out spans")
    write_csv(output / "pca_heldout_fidelity_by_family.csv", all_rows)
    write_csv(output / "pca_heldout_fidelity.csv", summary)
    return {
        "spans": len(all_rows),
        "matmul_reconstruction_hash_matches": sum(
            x["receiver_reconstruction_hash_matches"] for x in all_rows
        ),
        "either_summation_hash_matches": sum(
            x["receiver_reconstruction_hash_matches"]
            or x["alternate_summation_receiver_hash_matches"]
            for x in all_rows
        ),
    }
