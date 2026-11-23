"""Recompute the secondary retrieval comparisons from the archived measurements."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

import binary_codec
import routing


HERE = Path(__file__).resolve().parent
S3 = HERE.parents[1] / "evidence" / "Supplement_S3.zip"
S3_HASH = "0d68def998960697a83cf68fad2f7cd89c7d04618b47a229b3db9bee5bdb1005"
CHECKSUMS = {
    "codec_development.json": "2a6acc5da7c7cf955a4e84d6c4ddbc98098ff86d828438cbef02af54bc747e47",
    "two_means.bin": "0a54d4863bc4b15cde343dec7682f692b893d25c5cc8051a119cdf254ec02364",
    "codec_observations.npz": "2a39392f329e14e0e04d3b67ad8b72c9d1a3d9e456cf3a7de4cad01fda0066cc",
}
SIZES = {"arguana": 989, "bright": 103, "scifact": 300}
DECODED = {
    "arguana": "12776672c917bb7d8223182ca4ac280389d9e7fe1f7ee43d104f9d768c471893",
    "bright": "e05349e00796395a9a01a370791323d5adfba7c572772d9ff9feb073813ecea3",
    "scifact": "cc6777c98a5c6790f4982d8da51fbd3302f61c11995be76f00781f74456d1325",
}


def require(value, detail):
    if not value:
        raise ValueError(detail)


def checked_bytes(path, expected):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, f"Digest mismatch: {path.name}")
    return data


def scores(rows, count):
    require(rows.shape == (count, 3) and np.isfinite(rows).all(), "Incomplete metric rows")
    require(((0 <= rows) & (rows <= 1)).all(), "Invalid retrieval metrics")
    return rows.mean(axis=0)


def select(development, family, budget):
    candidates = [
        (key, value) for key, value in development.items()
        if value["family"] == family and value["bytes_max"] <= budget
        and (family != "text" or value["budget"] <= budget)
    ]
    require(bool(candidates), f"No feasible {family} for {budget} bytes")
    return min(candidates, key=lambda item: (-item[1]["metrics"][2], item[1]["bytes_mean"], item[0]))[0]


def envelope(archive):
    protocol = json.loads(archive.read("scientific/budget_envelope_protocol.json"))
    require(protocol["status"] == "FROZEN_BEFORE_HELDOUT", "Missing pre-test decision")
    development = json.loads(checked_bytes(HERE / "dev_results.json", protocol["inputs"]["dev_results.json"]))
    for family, count in (("text", 315), ("numeric", 28)):
        require(sum(v["family"] == family for v in development.values()) == count, "Incomplete grid")
    test = json.loads(archive.read("arguana/test_results.json"))
    reference = json.loads(archive.read("scientific/budget_envelope_test_results.json"))
    require(reference["status"] == "completed" and reference["queries"] == 989, "Incomplete test")
    output = []
    with np.load(io.BytesIO(archive.read("arguana/test_metrics.npz")), allow_pickle=False) as observations:
        for budget, prior in zip((32, 64, 128, 256, 512), reference["tradeoffs"], strict=True):
            frozen = protocol["selection"][str(budget)]
            numeric, text = (select(development, family, budget) for family in ("numeric", "text"))
            require((numeric, text) == (frozen["numeric_key"], frozen["text_key"]), "Selection changed")
            key_list = [numeric, text, frozen["nominal_text_key"]]
            require(key_list == [prior["numeric_key"], prior["text_key"], prior["nominal_text_key"]]
                    and prior["budget_bytes"] == budget, "Condition identity changed")
            result = {"cap": budget}
            for name, key in zip(("numeric", "text", "nominal_text"), key_list):
                mean = scores(observations[key], 989)
                np.testing.assert_allclose(mean, prior[name + "_metrics"], atol=1e-12, rtol=0)
                cost = test[key]["bytes_mean"]
                require(0 < cost <= test[key]["bytes_max"] <= budget, "Invalid packet size")
                if name != "nominal_text":
                    session = 989 * cost + test[key]["setup_bytes"]
                    np.testing.assert_allclose(session, prior[name + "_session_bytes_without_common_assets"])
                result[name] = {"key": key, "ndcg10": float(mean[2]), "mean_bytes": cost}
            diff = observations[numeric][:, 2] - observations[text][:, 2]
            for name, comparison in (("wins", diff > 0), ("losses", diff < 0), ("ties", diff == 0)):
                require(int(comparison.sum()) == prior["query_" + name], "Query comparison mismatch")
            output.append(result)
    return output


def fitted(archive):
    originals = {name: checked_bytes(HERE / name, h) for name, h in CHECKSUMS.items()}
    candidates = json.loads(originals["codec_development.json"])
    require(set(candidates) == set(binary_codec.KINDS), "Incomplete codec grid")
    require(all(v["n_queries"] == 412 for v in candidates.values()), "Incomplete development split")
    winner = min(candidates, key=lambda k: (-candidates[k]["metrics"][2], candidates[k]["wire_bytes_mean"], k))
    codebook = binary_codec.load_asset(originals["two_means.bin"])
    require(codebook.name == winner == "raw_two_means", "Codebook does not match development choice")
    setup = len(originals["two_means.bin"])
    output = {}
    with np.load(io.BytesIO(originals["codec_observations.npz"]), allow_pickle=False) as bank:
        for dataset, count in SIZES.items():
            packet_rows = bank[dataset + "_packets"]
            require(packet_rows.shape == (count, 56) and packet_rows.dtype == np.uint8, "Packet coverage")
            decoded = np.stack([binary_codec.decode(row.tobytes(), codebook) for row in packet_rows])
            require(hashlib.sha256(decoded.tobytes()).hexdigest() == DECODED[dataset], "Receiver decode mismatch")
            if dataset == "bright":
                rows = np.load(io.BytesIO(archive.read("bright/metrics_numeric_supplement.npy")), allow_pickle=False)
                ref = json.loads(archive.read("bright/results.json"))
                expected = [ref["summaries"]["numeric_supplement"]["mean"][k] for k in ("hit1","mrr10","ndcg10")]
                require(ref["setup_bytes"]["numeric_supplement"] == setup, "Fitted setup mismatch")
            else:
                rows = bank[dataset + "_metrics"]
                ref = json.loads(archive.read("scientific/supplement_" + dataset + "_results.json"))
                expected = ref["metrics"]
                require(ref["execution_failures"] == 0 and ref["queries"] == count and ref["setup_bytes"] == setup,
                        "Fitted evaluation incomplete")
            mean = scores(rows, count)
            np.testing.assert_allclose(mean, expected, atol=1e-12, rtol=0)
            wire = packet_rows.nbytes / count
            output[dataset] = {"queries": count, "ndcg10": float(mean[2]), "wire_bytes": wire,
                               "setup_bytes": setup, "amortized_bytes": wire + setup / count}
    return output


def main():
    with zipfile.ZipFile(io.BytesIO(checked_bytes(S3, S3_HASH))) as archive:
        results = {"budget_envelope": envelope(archive), "fitted_codec": fitted(archive), "routing": routing.run(archive)}
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
