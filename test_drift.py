"""Smoke test for config drift detection.

Constructs CodebaseRAG once (so we only pay the embedding-load cost a single
time), then mutates the in-memory `config_snapshot` to simulate four
scenarios:
  1. baseline (snapshot matches live config) - expects no drift,
  2. warning-severity drift (supported_extensions differ),
  3. critical-severity drift (embedding model differs),
  4. drift_status_text() formats critical drift correctly.

The on-disk `metadata.json` is backed up before the test and restored
afterwards, so the user's actual index is left exactly as it was.
"""

import json
import shutil
import sys
from pathlib import Path

from config import load_config
from rag_manager import CodebaseRAG, DRIFT_CRITICAL, DRIFT_WARNING


def main() -> int:
    cfg = load_config()
    metadata_file = cfg.storage_path / "metadata.json"

    if not metadata_file.exists():
        print(
            f"[test] FAIL: {metadata_file} does not exist. "
            "Run an indexing pass first (start the server once, or "
            "invoke update_codebase_index)."
        )
        return 1

    backup_path = metadata_file.with_suffix(".json.bak")
    shutil.copy2(metadata_file, backup_path)
    print(f"[test] Backed up {metadata_file.name} -> {backup_path.name}")

    try:
        print("[test] Constructing CodebaseRAG (one embedding-model load)...")
        rag = CodebaseRAG(
            codebase_path=str(cfg.codebase_path),
            rag_storage_path=str(cfg.storage_path),
            supported_extensions=cfg.supported_extensions,
            embedding_model_name=cfg.embedding.model_name,
            collection_name=cfg.collection_name,
        )
        current_snapshot = rag._build_config_snapshot()

        # ----- 1. baseline: snapshot matches live config, no drift -----
        rag.metadata["config_snapshot"] = dict(current_snapshot)
        drift = rag.check_config_drift()
        if drift is not None:
            print(f"[test] FAIL: matching snapshot reported drift: {drift}")
            return 2
        print("[test] OK [1/4] no drift when snapshot matches live config")

        # ----- 2. warning severity: extensions changed -----
        warning_snapshot = dict(current_snapshot)
        warning_snapshot["supported_extensions"] = sorted(
            list(current_snapshot["supported_extensions"]) + [".faketestext"]
        )
        rag.metadata["config_snapshot"] = warning_snapshot
        drift = rag.check_config_drift()
        if drift is None:
            print("[test] FAIL: no drift detected for changed extensions")
            return 3
        if drift["severity"] != DRIFT_WARNING:
            print(
                f"[test] FAIL: expected '{DRIFT_WARNING}', got '{drift['severity']}'"
            )
            return 4
        if "supported_extensions" not in drift["changes"]:
            print(
                f"[test] FAIL: missing 'supported_extensions' in changes: "
                f"{drift['changes']}"
            )
            return 5
        print("[test] OK [2/4] warning drift detected for changed extensions")

        # ----- 3. critical severity: embedding model changed -----
        critical_snapshot = dict(current_snapshot)
        critical_snapshot["embedding_model_name"] = "fake/other-model-v999"
        rag.metadata["config_snapshot"] = critical_snapshot
        drift = rag.check_config_drift()
        if drift is None:
            print("[test] FAIL: no drift detected for changed embedding model")
            return 6
        if drift["severity"] != DRIFT_CRITICAL:
            print(
                f"[test] FAIL: expected '{DRIFT_CRITICAL}', got '{drift['severity']}'"
            )
            return 7
        if "embedding_model_name" not in drift["changes"]:
            print(
                f"[test] FAIL: missing 'embedding_model_name' in changes: "
                f"{drift['changes']}"
            )
            return 8
        print("[test] OK [3/4] critical drift detected for changed embedding model")

        # ----- 4. drift_status_text formats critical drift -----
        text = rag.drift_status_text()
        if "CRITICAL" not in text or "embedding_model_name" not in text:
            print(f"[test] FAIL: drift_status_text missing key info:\n{text}")
            return 9
        print("[test] OK [4/4] drift_status_text formats critical drift correctly")

        print("\n[test] === SUCCESS: drift detection works as expected ===")
        return 0
    finally:
        shutil.copy2(backup_path, metadata_file)
        backup_path.unlink()
        print(f"[test] Restored {metadata_file.name} from backup")


if __name__ == "__main__":
    sys.exit(main())
