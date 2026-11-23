"""Generate calibration/development families only; heldout creation is a later frozen step."""
import argparse
from pathlib import Path
import secrets

from common import Stage, read_json, write_json, sha
from private_tasks import generate_qa_pair, generate_navigation_pair, validate_split_isolation


def build(output, calibration=32, development=8):
    stage = Stage(output, "development_data", requested=2 * (calibration + development))
    try:
        seed = secrets.token_hex(32)
        root = Path(__file__).resolve().parents[1]
        unavailable = set()
        # Exact previous lexical secrets remain excluded from fresh QA values.
        for path in (root / "vector_solution/studies").glob("*/study/private_cases.json"):
            doc = read_json(path)
            for row in doc["cases"]:
                unavailable.update((row["target"], row["alternate"]))
        forbidden = read_json(root / "vector_solution/inputs/final_frozen_v1/forbidden_values.json")
        def strings(value):
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [s for item in value for s in strings(item)]
            if isinstance(value, dict):
                return [s for item in value.values() for s in strings(item)]
            return []
        unavailable.update(s for s in strings(forbidden) if len(s) == 16 and s.isalpha() and s.isupper())
        initial = sorted(unavailable)
        all_families = []
        for split, count in (("calibration", calibration), ("development", development)):
            families = []
            for task, generator in (("qa", generate_qa_pair), ("navigation", generate_navigation_pair)):
                for index in range(count):
                    kwargs = {"rows": 4 + index % 3} if task == "qa" else {"size": 5 + index % 2}
                    family = generator(seed, split=split, family_id=f"{split}_{task}_{index:05}",
                                       excluded_values=unavailable, **kwargs)
                    unavailable.update(family["exclusion_inventory"])
                    families.append(family)
                    all_families.append(family)
                    stage.update(n_completed=len(all_families))
            write_json(stage.directory / (split + ".json"), {"split": split, "families": families})
        validate_split_isolation(all_families)
        write_json(stage.directory / "lineage.json", {"private_master_seed": seed,
            "generator_sha256": sha(Path(__file__).with_name("private_tasks.py")),
            "initial_exclusions": initial, "all_exclusions": sorted(unavailable),
            "calibration_families_per_task": calibration, "development_families_per_task": development,
            "heldout_created": False})
        stage.finish()
    except BaseException as error:
        stage.fail(error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=32)
    parser.add_argument("--development", type=int, default=8)
    args = parser.parse_args()
    build(args.output, args.calibration, args.development)
