"""Fail-fast artifact lineage locks for formal BiVES-CXR runs."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
from typing import Any

import torch
import yaml

from .calibration import probabilities_from_evidence
from .decoder import DECODER_PARAMETER_NAMES
from .interventions import CONTROL_PROTOCOL_VERSION
from .metrics import classification_metrics


RUN_LOCK_FORMAT_VERSION = 3
MODEL_SNAPSHOT_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "tokenizer_config.json", "tokenizer.json",
    "chat_template.json", "chat_template.jinja", "special_tokens_map.json",
    "added_tokens.json", "merges.txt", "vocab.json", "vocab.txt", "spiece.model",
)
PROTECTED_SOURCE_ROOTS = (
    "bives_cxr",
    "scripts",
    "configs/bives_cxr",
    "tests",
    "docs",
)
ROOT_EXECUTABLE_SUFFIXES = {
    ".py", ".pyi", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".ps1", ".bat", ".cmd", ".pth",
}
ROOT_EXECUTABLE_NAMES = {"sitecustomize.py", "usercustomize.py"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolved_config_text(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def resolved_config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(resolved_config_text(config).encode("utf-8")).hexdigest()


def base_model_lock_files(model_path: str | Path) -> tuple[Path, Path]:
    root = Path(model_path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"base model config is missing: {config_path}")
    candidates = (
        root / "model.safetensors.index.json",
        root / "model.safetensors",
    )
    weight_lock = next((path for path in candidates if path.is_file()), None)
    if weight_lock is None:
        raise FileNotFoundError(
            f"base model has no safetensors index or single weight file: {root}"
        )
    return config_path, weight_lock


def base_model_snapshot(model_path: str | Path) -> dict[str, Any]:
    """Hash every local artifact needed to reproduce Qwen3.5 visual loading."""

    root = Path(model_path)
    config_path, weight_lock = base_model_lock_files(root)
    files: set[Path] = {config_path}
    if weight_lock.name.endswith(".index.json"):
        index = json.loads(weight_lock.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model safetensors index must contain a non-empty weight_map")
        files.add(weight_lock)
        for shard in sorted(set(str(value) for value in weight_map.values())):
            shard_path = root / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"model shard referenced by index is missing: {shard_path}")
            files.add(shard_path)
    else:
        files.add(weight_lock)
    for name in MODEL_SNAPSHOT_FILES:
        candidate = root / name
        if candidate.is_file():
            files.add(candidate)
    file_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): file_sha256(path)
        for path in sorted(files)
    }
    return {
        "root": str(root),
        "files": file_hashes,
        "sha256": canonical_json_sha256(file_hashes),
    }


def _protected_source_inventory(root: Path) -> set[str]:
    """Return executable/configuration files that must be manifest-listed.

    A source-only deployment cannot rely on ``git status`` to catch an ignored
    or injected Python module.  These roots comprise every active code/config
    surface; historical material remains outside this protected inventory.
    """

    protected: set[str] = set()
    for relative_root in PROTECTED_SOURCE_ROOTS:
        directory = root / relative_root
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            if path.is_file() or path.is_symlink():
                protected.add(relative.as_posix())
    for path in root.iterdir():
        if not (path.is_file() or path.is_symlink()):
            continue
        if path.name in ROOT_EXECUTABLE_NAMES or path.suffix.lower() in ROOT_EXECUTABLE_SUFFIXES:
            protected.add(path.relative_to(root).as_posix())
    return protected


def _validate_protected_source_inventory(root: Path, files: dict[str, str]) -> None:
    """Reject extra, unlisted, or symlinked active files before a formal run."""

    listed = {str(path).replace("\\", "/") for path in files}
    actual = _protected_source_inventory(root)
    extra = sorted(actual - listed)
    if extra:
        raise ValueError(
            "source snapshot has unlisted active files: " + ", ".join(extra[:8])
        )
    symlinked = sorted(
        rel for rel in actual if (root / rel).is_symlink()
    )
    if symlinked:
        raise ValueError(
            "source snapshot does not permit active symlinks: "
            + ", ".join(symlinked[:8])
        )
def _git_tracked_source_snapshot(root: Path, *, require_clean: bool) -> dict[str, Any] | None:
    probe = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        return None
    repository = Path(probe.stdout.strip())
    status = subprocess.run(["git", "-C", str(repository), "status", "--porcelain"], capture_output=True, text=True, check=False)
    if require_clean and status.stdout.strip():
        raise ValueError("formal runs require a clean Git worktree")
    listed = subprocess.run(["git", "-C", str(repository), "ls-files", "-z"], capture_output=True, check=True)
    files: dict[str, str] = {}
    for raw in listed.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8")
        path = repository / rel
        if path.is_file():
            files[rel.replace("\\", "/")] = file_sha256(path)
    commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    _validate_protected_source_inventory(repository, files)
    return {"kind": "git", "git_commit": commit, "files": files, "tree_sha256": canonical_json_sha256(files)}


def build_source_snapshot(*, root: str | Path | None = None, require_clean: bool = True) -> dict[str, Any]:
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    git_snapshot = _git_tracked_source_snapshot(root_path, require_clean=require_clean)
    if git_snapshot is not None:
        return git_snapshot
    marker = root_path / ".bives_source_manifest.json"
    if not marker.is_file():
        raise FileNotFoundError("source-only deployment requires .bives_source_manifest.json")
    snapshot = json.loads(marker.read_text(encoding="utf-8"))
    if snapshot.get("kind") != "source_archive" or not isinstance(snapshot.get("files"), dict):
        raise ValueError("source snapshot manifest is malformed")
    files = snapshot["files"]
    for rel, expected in files.items():
        path = root_path / rel
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"source snapshot mismatch: {rel}")
    _validate_protected_source_inventory(root_path, files)
    if canonical_json_sha256(files) != snapshot.get("tree_sha256"):
        raise ValueError("source snapshot tree SHA256 does not match file inventory")
    return snapshot


def build_git_archive_source_snapshot(root: str | Path | None = None) -> dict[str, Any]:
    """Hash emitted ``git archive`` bytes, including attribute-driven EOL conversion."""

    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    probe = subprocess.run(["git", "-C", str(root_path), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        raise ValueError("archive source snapshot requires a Git checkout")
    repository = Path(probe.stdout.strip())
    status = subprocess.run(["git", "-C", str(repository), "status", "--porcelain"], capture_output=True, text=True, check=False)
    if status.stdout.strip():
        raise ValueError("archive source snapshot requires a clean Git worktree")
    commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    archive = subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", "HEAD"],
        capture_output=True,
        check=True,
    ).stdout
    files: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"archive member is unreadable: {member.name}")
            files[member.name] = hashlib.sha256(extracted.read()).hexdigest()
    _validate_protected_source_inventory(repository, files)
    return {"kind": "source_archive", "git_commit": commit, "files": files, "tree_sha256": canonical_json_sha256(files)}


def build_run_lock(
    config: dict[str, Any],
    *,
    git_commit: str,
    source_snapshot: dict[str, Any] | None = None,
    dataset_lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not str(git_commit).strip() or str(git_commit).strip().lower() == "unknown":
        raise ValueError(
            "formal run lock requires a known Git commit; set BIVES_GIT_COMMIT "
            "or deploy a .bives_source_commit marker"
        )
    data = config["data"]
    model = config["model"]
    statement = model["statement_embeddings"]
    cache_path = Path(statement["path"])
    if not cache_path.is_file():
        raise FileNotFoundError(f"statement cache is missing: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    vocabulary_sha = str(
        cache.get("ontology", {}).get("vocabulary_sha256", "")
    )
    if not vocabulary_sha:
        raise ValueError("statement cache is missing ontology vocabulary_sha256")
    model_snapshot = base_model_snapshot(model["path"])
    manifests: dict[str, str] = {}
    for split, key in (
        ("train", "train_manifest"),
        ("val", "val_manifest"),
        ("calibration", "calibration_manifest"),
        ("test", "test_manifest"),
    ):
        path = Path(str(data.get(key, "")))
        if not path.is_file():
            raise FileNotFoundError(f"locked {split} manifest is missing: {path}")
        manifests[split] = file_sha256(path)
    control_seed = config.get("evaluation", {}).get("control_seed")
    if control_seed is None:
        raise ValueError("evaluation.control_seed is required for a formal run lock")
    if dataset_lock is None:
        raise ValueError("formal run lock requires a validated joint dataset lock")
    if dataset_lock.get("status") != "pass":
        raise ValueError("formal run lock requires a passing dataset lock")
    if source_snapshot is None:
        source_snapshot = {"kind": "legacy_test", "git_commit": str(git_commit)}
    return {
        "format_version": RUN_LOCK_FORMAT_VERSION,
        "git_commit": str(git_commit),
        "resolved_config_sha256": resolved_config_sha256(config),
        "source_snapshot": source_snapshot,
        "source_tree_sha256": source_snapshot.get("tree_sha256"),
        "base_model_snapshot": model_snapshot,
        "base_model_snapshot_sha256": model_snapshot["sha256"],
        "statement_cache_sha256": file_sha256(cache_path),
        "statement_vocabulary_sha256": vocabulary_sha,
        "manifest_sha256": manifests,
        "dataset_lock_sha256": canonical_json_sha256(dataset_lock),
        "control_protocol_version": CONTROL_PROTOCOL_VERSION,
        "evaluation_control_seed": int(control_seed),
    }


def validate_checkpoint_run_lock(
    checkpoint: dict[str, Any],
    *,
    current_git_commit: str,
) -> dict[str, Any]:
    run_lock = checkpoint.get("run_lock")
    if not isinstance(run_lock, dict):
        raise ValueError("checkpoint is missing run_lock")
    if run_lock.get("format_version") != RUN_LOCK_FORMAT_VERSION:
        raise ValueError("checkpoint run_lock format_version is unsupported")
    expected_hash = canonical_json_sha256(run_lock)
    if checkpoint.get("run_lock_sha256") != expected_hash:
        raise ValueError("checkpoint run_lock_sha256 does not match its run_lock")
    if str(run_lock.get("git_commit")) != str(current_git_commit):
        raise ValueError(
            "current evaluator Git commit does not match checkpoint run_lock"
        )
    source_snapshot = run_lock.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise ValueError("checkpoint run_lock is missing source snapshot")
    if source_snapshot.get("kind") != "legacy_test":
        current_source = build_source_snapshot(require_clean=True)
        if current_source.get("tree_sha256") != source_snapshot.get("tree_sha256"):
            raise ValueError("current source tree does not match run_lock")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing resolved config")
    if resolved_config_sha256(config) != run_lock.get("resolved_config_sha256"):
        raise ValueError("checkpoint config does not match resolved_config_sha256")
    current_model_snapshot = base_model_snapshot(config["model"]["path"])
    if current_model_snapshot != run_lock.get("base_model_snapshot"):
        raise ValueError("base model snapshot does not match run_lock")
    evaluation_seed = config.get("evaluation", {}).get("control_seed")
    if int(evaluation_seed) != int(run_lock.get("evaluation_control_seed")):
        raise ValueError("evaluation control seed does not match run_lock")
    if run_lock.get("control_protocol_version") != CONTROL_PROTOCOL_VERSION:
        raise ValueError("control protocol version does not match active evaluator")
    return run_lock


def validate_calibrated_release_chain(
    *,
    checkpoint_path: str | Path,
    checkpoint: dict[str, Any],
    calibration: dict[str, Any],
    statement_cache_path: str | Path,
    test_manifest_path: str | Path,
    current_git_commit: str,
    calibration_artifact_path: str | Path | None = None,
    dataset_lock_path: str | Path | None = None,
    data_root: str | Path | None = None,
    dataset_manifests: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    run_lock = validate_checkpoint_run_lock(
        checkpoint,
        current_git_commit=current_git_commit,
    )
    checks = {
        "base checkpoint": (
            file_sha256(checkpoint_path),
            calibration.get("base_checkpoint_sha256"),
        ),
        "run lock": (
            checkpoint.get("run_lock_sha256"),
            calibration.get("run_lock_sha256"),
        ),
        "statement cache": (
            file_sha256(statement_cache_path),
            run_lock.get("statement_cache_sha256"),
        ),
        "calibration statement cache": (
            calibration.get("statement_cache_sha256"),
            run_lock.get("statement_cache_sha256"),
        ),
        "statement vocabulary": (
            calibration.get("statement_vocabulary_sha256"),
            run_lock.get("statement_vocabulary_sha256"),
        ),
        "locked test manifest": (
            file_sha256(test_manifest_path),
            run_lock.get("manifest_sha256", {}).get("test"),
        ),
        "selected best step": (
            int(checkpoint.get("step", -1)),
            int(calibration.get("selected_best_step", -2)),
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError(
            "locked-test artifact chain mismatch: " + ", ".join(mismatches)
        )
    if calibration_artifact_path is not None:
        payload = dict(calibration)
        declared = payload.pop("canonical_artifact_sha256", None)
        if declared != canonical_json_sha256(payload):
            raise ValueError("calibration artifact canonical SHA256 does not match")
    expected_manifest = run_lock.get("manifest_sha256", {}).get("calibration")
    if calibration.get("calibration_manifest_sha256") != expected_manifest:
        raise ValueError("calibration manifest does not match run_lock")
    if calibration.get("control_protocol_version") != run_lock.get("control_protocol_version"):
        raise ValueError("calibration control protocol does not match run_lock")
    if int(calibration.get("evaluation_control_seed", -1)) != int(run_lock.get("evaluation_control_seed")):
        raise ValueError("calibration evaluation control seed does not match run_lock")
    if calibration.get("calibration_algorithm") != "monotone_decoder_lbfgs_v1":
        raise ValueError("calibration algorithm/version is unsupported")
    for field in (
        "uncalibrated_decoder_parameters",
        "calibrated_decoder_parameters",
    ):
        values = calibration.get(field)
        if not isinstance(values, dict):
            raise ValueError(f"calibration artifact is missing {field}")
        for name in DECODER_PARAMETER_NAMES:
            value = float(values.get(name, float("nan")))
            if not math.isfinite(value) or not 1e-4 <= value <= 1e4:
                raise ValueError(f"calibration {field}.{name} is not finite and bounded")
    checkpoint_parameters = {
        name: float(checkpoint["bives_head"][f"decoder.{name}"].detach().cpu())
        for name in DECODER_PARAMETER_NAMES
    }
    if any(
        not math.isclose(
            checkpoint_parameters[name],
            float(calibration["uncalibrated_decoder_parameters"][name]),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        for name in checkpoint_parameters
    ):
        raise ValueError("calibration uncalibrated decoder parameters do not match checkpoint")
    for field in ("calibration_pre_nll", "calibration_post_nll"):
        if not math.isfinite(float(calibration.get(field, float("nan")))):
            raise ValueError(f"calibration artifact has invalid {field}")
    if float(calibration["calibration_post_nll"]) > float(calibration["calibration_pre_nll"]) + 1e-8:
        raise ValueError("calibration NLL regressed")
    prediction_ref = calibration.get("calibration_predictions_file")
    if not isinstance(prediction_ref, str) or not prediction_ref.strip():
        raise ValueError("calibration artifact is missing calibration_predictions_file")
    prediction_path = Path(prediction_ref)
    if not prediction_path.is_absolute() and calibration_artifact_path is not None:
        prediction_path = Path(calibration_artifact_path).resolve().parent / prediction_path
    if not prediction_path.is_file():
        raise FileNotFoundError(f"calibration predictions are missing: {prediction_path}")
    if file_sha256(prediction_path) != calibration.get("calibration_predictions_sha256"):
        raise ValueError("calibration prediction SHA256 does not match")
    recomputed_pre_nll = _calibration_prediction_nll(
        prediction_path,
        calibration["uncalibrated_decoder_parameters"],
    )
    recomputed_post_nll = _calibration_prediction_nll(
        prediction_path,
        calibration["calibrated_decoder_parameters"],
    )
    for field, actual in (
        ("calibration_pre_nll", recomputed_pre_nll),
        ("calibration_post_nll", recomputed_post_nll),
    ):
        if not math.isclose(
            float(calibration[field]), actual, rel_tol=0.0, abs_tol=2e-6
        ):
            raise ValueError(
                f"{field} does not match locked calibration prediction evidence"
            )
    if dataset_lock_path is not None:
        from .dataset_lock import validate_dataset_lock
        required_splits = ("train", "val", "calibration", "test")
        if not isinstance(dataset_manifests, dict):
            raise ValueError("dataset lock validation requires explicit four-split manifests")
        missing = [split for split in required_splits if split not in dataset_manifests]
        if missing:
            raise ValueError(
                "dataset lock validation is missing manifests: " + ", ".join(missing)
            )
        if data_root is None:
            raise ValueError("dataset lock validation requires an explicit data_root")
        manifests = {split: dataset_manifests[split] for split in required_splits}
        lock = validate_dataset_lock(dataset_lock_path, manifests, data_root=data_root)
        if canonical_json_sha256(lock) != run_lock.get("dataset_lock_sha256"):
            raise ValueError("dataset lock does not match run_lock")
    return run_lock


def _calibration_prediction_nll(
    prediction_path: str | Path,
    decoder_parameters: dict[str, Any],
) -> float:
    """Rebuild decoder probabilities from immutable calibration evidence rows."""

    evidence_pos: list[float] = []
    evidence_neg: list[float] = []
    targets: list[int] = []
    with Path(prediction_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                positive = float(row["evidence_pos"])
                negative = float(row["evidence_neg"])
                target = int(row["target"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid calibration prediction row {line_number}"
                ) from exc
            if not math.isfinite(positive) or not math.isfinite(negative):
                raise ValueError(
                    f"calibration prediction row {line_number} has non-finite evidence"
                )
            if target not in (0, 1, 2, 3):
                raise ValueError(
                    f"calibration prediction row {line_number} has invalid target"
                )
            evidence_pos.append(positive)
            evidence_neg.append(negative)
            targets.append(target)
    if not targets:
        raise ValueError("calibration predictions contain no rows")
    probabilities = probabilities_from_evidence(
        torch.tensor(evidence_pos, dtype=torch.float32),
        torch.tensor(evidence_neg, dtype=torch.float32),
        torch.tensor(float(decoder_parameters["tau_a"]), dtype=torch.float32),
        torch.tensor(float(decoder_parameters["tau_p"]), dtype=torch.float32),
        torch.tensor(float(decoder_parameters["uncertainty_mass"]), dtype=torch.float32),
    )
    return float(
        classification_metrics(
            probabilities.detach().cpu(), torch.tensor(targets, dtype=torch.long)
        )["nll"]
    )
