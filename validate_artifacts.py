"""CPU-only проверка целостности опубликованных артефактов (стандартная библиотека)."""
import ast
import hashlib
import json
import math
import struct
from statistics import mean

from common import ROOT, read_jsonl
from checks import check_answer, self_test
from summarize import METRICS, bootstrap_delta, summarize_judge


def main():
    self_test()
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    groups = {}
    for split, meta in manifest["splits"].items():
        path = ROOT / "data" / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
        records = read_jsonl(path)
        assert len(records) == meta["rows"]
        groups[split] = {r["group"] for r in records}
        assert len(groups[split]) == meta["groups"]
    assert not (groups["train"] & groups["dev"] or groups["train"] & groups["test"] or groups["dev"] & groups["test"])
    before, after = [read_jsonl(ROOT / "outputs" / f"{v}.jsonl") for v in ("before", "after")]
    expected = read_jsonl(ROOT / "data/test.jsonl")
    assert [r["id"] for r in before] == [r["id"] for r in after] == [r["id"] for r in expected]
    for record in before + after:
        for key, value in check_answer(record["answer"]).items():
            assert record["checks"][key] == value, (record["id"], key)
        assert record["checks"]["urgency_correct"] == (record["checks"]["predicted_urgency"] == record["expected_urgency"])
    saved = json.loads((ROOT / "outputs/metrics.json").read_text())
    for key, row in zip(METRICS, saved):
        assert row["metric"] == key
        for variant, records in (("before", before), ("after", after)):
            assert math.isclose(row[variant], mean(int(r["checks"][key]) for r in records))
        lo, hi = bootstrap_delta(before, after, key)
        assert math.isclose(row["delta_ci_low_pp"], 100 * lo, abs_tol=1e-10)
        assert math.isclose(row["delta_ci_high_pp"], 100 * hi, abs_tol=1e-10)
    assert summarize_judge() == json.loads((ROOT / "outputs/judge_summary.json").read_text())
    weights = ROOT / "outputs/adapter/adapter_model.safetensors"
    with weights.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    assert max(v["data_offsets"][1] for v in tensors) == weights.stat().st_size - 8 - size
    count = sum(math.prod(v["shape"]) for v in tensors)
    assert count == json.loads((ROOT / "outputs/train_summary.json").read_text())["trainable_parameters"]
    assert (ROOT / "output/pdf/REPORT.pdf").read_bytes().startswith(b"%PDF-")
    print(f"OK: данные, split, {len(before) + len(after)} ответов, метрики, bootstrap, судья и {count:,} параметров адаптера.")


if __name__ == "__main__":
    main()
