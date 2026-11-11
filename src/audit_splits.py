#!/usr/bin/env python3
"""Audit train/test splits for target leakage.

Implements the per-fold audit described in Appendix B of
"Towards Benchmarking Agentic Data Scientists" (AutoAI-FM @ IJCAI-ECAI 2026).
Every fold is checked for five conditions:

  1. target_in_test   the target column itself is present in the test file
  2. single_feature   the target is exactly reconstructed by one feature
                      (equality or an exact affine map for numeric targets,
                      a one-to-one relabelling for classification targets)
  3. additive_pair    the target equals the sum of two features
  4. schema           train and test files do not have the same feature columns
                      (plus a cross-fold comparison of schemas within a dataset)
  5. row_overlap      feature rows that appear in both train and test

Checks 2 and 3 run on the training file and, when held-out labels are
available (a labels file next to the test file, or a leaked target column),
on the test file as well. Checks 1, 2, 3 and 5 are counted as failures,
schema differences as warnings.

Usage
-----
Audit every fold under a directory (any layout; train files are found by
name and paired with the matching test file):

    python src/audit_splits.py --root data/folds --out-json results/audit.json

Audit a single fold:

    python src/audit_splits.py --train fold1/train.csv --test fold1/test.csv \
        --test-labels fold1/test_y.csv --target count

The exit code is 1 when any fold fails, so the script can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Target names of the datasets in the original 12-dataset suite, used only when
# the target cannot be inferred from the files themselves.
KNOWN_TARGETS = [
    "target", "TARGET", "Target", "count", "Activity", "Cover_Type",
    "SeriousDlqin2yrs", "Choice", "SalePrice", "y", "revenue", "price_doc",
]
LABEL_FILE_NAMES = ["test_y.csv", "y_test.csv", "test_labels.csv", "test_target.csv"]
ID_NAMES = {"id"}

RTOL = 1e-6
ATOL = 1e-8
R2_EXACT = 1 - 1e-9          # an affine fit at least this good counts as exact
MAX_PAIR_CANDIDATES = 10_000  # cap on candidate pairs verified per fold


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class Finding:
    check: str
    severity: str           # "FAIL" or "WARN"
    split: str              # "train", "test" or "train/test"
    detail: str
    r2: Optional[float] = None


@dataclass
class FoldReport:
    dataset: str
    fold: str
    train_path: str
    test_path: str
    labels_path: Optional[str]
    target: Optional[str]
    task: Optional[str]
    n_train: int = 0
    n_test: int = 0
    n_train_cols: int = 0
    n_test_cols: int = 0
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def status(self) -> str:
        sev = {f.severity for f in self.findings}
        return "FAIL" if "FAIL" in sev else ("WARN" if "WARN" in sev else "PASS")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def ci_lookup(columns, name: str) -> Optional[str]:
    """Return the column matching `name` case-insensitively, if any."""
    if name in columns:
        return name
    lowered = {c.lower(): c for c in columns}
    return lowered.get(name.lower())


def id_columns(df: pd.DataFrame, extra: set[str]) -> list[str]:
    names = {n.lower() for n in ID_NAMES | extra}
    return [c for c in df.columns if c.lower() in names]


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float("nan")
    return 1 - ss_res / ss_tot


def resolve_target(train: pd.DataFrame, test: pd.DataFrame,
                   labels: Optional[pd.DataFrame], explicit: Optional[str],
                   extra_ids: set[str]) -> tuple[Optional[str], str]:
    """Find the target column in the training file. Returns (name, how)."""
    if explicit:
        col = ci_lookup(train.columns, explicit)
        return (col, "--target") if col else (None, f"--target '{explicit}' not in train")

    if labels is not None:
        label_cols = [c for c in labels.columns if c not in id_columns(labels, extra_ids)]
        if len(label_cols) == 1:
            col = ci_lookup(train.columns, label_cols[0])
            if col:
                return col, "labels file"

    test_lower = {c.lower() for c in test.columns}
    train_only = [c for c in train.columns if c.lower() not in test_lower]
    if len(train_only) == 1:
        return train_only[0], "only column missing from test"

    for name in KNOWN_TARGETS:
        col = ci_lookup(train.columns, name)
        if col:
            return col, "known target name"

    return None, "could not infer target; pass --target"


def infer_task(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    return "regression" if y.nunique(dropna=True) > 20 else "classification"


def numeric_matrix(df: pd.DataFrame, rows: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Numeric feature columns that are finite on `rows`, as a float matrix."""
    cols, mats = [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(float)
        if not pd.api.types.is_numeric_dtype(s):
            continue
        v = s.to_numpy(dtype=float)[rows]
        if np.isfinite(v).all():
            cols.append(c)
            mats.append(v)
    if not mats:
        return [], np.empty((rows.sum(), 0))
    return cols, np.column_stack(mats)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def check_single_feature(X: pd.DataFrame, y: pd.Series, task: str, split: str) -> list[Finding]:
    out = []
    if task == "regression" and pd.api.types.is_numeric_dtype(y):
        yv = y.to_numpy(dtype=float)
        rows = np.isfinite(yv)
        cols, M = numeric_matrix(X, rows)
        yv = yv[rows]
        if M.shape[1] == 0 or len(yv) < 3:
            return out
        equal = np.isclose(M, yv[:, None], rtol=RTOL, atol=ATOL).all(axis=0)
        for c in np.array(cols)[equal]:
            out.append(Finding("single_feature", "FAIL", split,
                               f"feature '{c}' equals the target", r2=1.0))
        # exact affine relation: |corr| == 1
        Mc = M - M.mean(axis=0)
        yc = yv - yv.mean()
        denom = np.sqrt((Mc ** 2).sum(axis=0) * (yc ** 2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = (Mc * yc[:, None]).sum(axis=0) / denom
        affine = (corr ** 2 >= R2_EXACT) & ~equal & (denom > 0)
        for c in np.array(cols)[affine]:
            out.append(Finding("single_feature", "FAIL", split,
                               f"target is an exact affine function of '{c}'", r2=1.0))
    else:
        # classification: one-to-one relabelling between a feature and the target
        yv = y.astype("string").fillna("<NA>")
        k = yv.nunique()
        for c in X.columns:
            f = X[c].astype("string").fillna("<NA>")
            if f.nunique() != k:
                continue
            pairs = pd.DataFrame({"f": f.values, "y": yv.values}).drop_duplicates()
            if len(pairs) == k:
                out.append(Finding("single_feature", "FAIL", split,
                                   f"feature '{c}' maps one-to-one onto the target"))
    return out


def check_additive_pair(X: pd.DataFrame, y: pd.Series, split: str,
                        seed: int = 0) -> tuple[list[Finding], list[str]]:
    """Find feature pairs (i, j) with x_i + x_j == y on every row.

    Uses random projections to shortlist candidate pairs in O(m log m),
    then verifies each candidate exactly on all rows.
    """
    notes: list[str] = []
    if not pd.api.types.is_numeric_dtype(y):
        return [], notes
    yv = y.to_numpy(dtype=float)
    rows = np.isfinite(yv)
    cols, M = numeric_matrix(X, rows)
    yv = yv[rows]
    skipped = sum(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns) - len(cols)
    if skipped:
        notes.append(f"{split}: {skipped} numeric column(s) with missing values skipped in pair search")
    m = M.shape[1]
    if m < 2 or len(yv) < 3:
        return [], notes

    rng = np.random.default_rng(seed)
    w = rng.standard_normal(len(yv))
    s = M.T @ w
    t = float(w @ yv)
    scale = float(np.abs(w) @ (np.abs(yv) + np.abs(M).max(axis=1) * 2 + 1))
    tol = 1e-6 * scale

    order = np.argsort(s)
    s_sorted = s[order]
    candidates = []
    for a in range(m):
        need = t - s[a]
        lo = np.searchsorted(s_sorted, need - tol, side="left")
        hi = np.searchsorted(s_sorted, need + tol, side="right")
        for b in order[lo:hi]:
            if b > a:
                candidates.append((a, b))
        if len(candidates) > MAX_PAIR_CANDIDATES:
            notes.append(f"{split}: more than {MAX_PAIR_CANDIDATES} candidate pairs; "
                         "only the first ones were verified")
            break

    out = []
    for a, b in candidates[:MAX_PAIR_CANDIDATES]:
        pred = M[:, a] + M[:, b]
        if np.isclose(pred, yv, rtol=RTOL, atol=ATOL).all():
            out.append(Finding("additive_pair", "FAIL", split,
                               f"'{cols[a]}' + '{cols[b]}' reproduces the target",
                               r2=r2_score(yv, pred)))
    return out, notes


def check_schema(train_feats: list[str], test: pd.DataFrame, target: Optional[str]) -> list[Finding]:
    out = []
    test_cols = [c for c in test.columns if not (target and c.lower() == target.lower())]
    tr, te = set(train_feats), set(test_cols)
    missing, extra = sorted(tr - te), sorted(te - tr)
    if missing:
        out.append(Finding("schema", "WARN", "train/test",
                           f"{len(missing)} train feature(s) missing from test: {missing[:10]}"))
    if extra:
        out.append(Finding("schema", "WARN", "train/test",
                           f"{len(extra)} test column(s) not in train: {extra[:10]}"))
    common = [c for c in train_feats if c in te]
    if not missing and not extra and common != [c for c in test_cols if c in tr]:
        out.append(Finding("schema", "WARN", "train/test", "column order differs between train and test"))
    return out


def row_hashes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Hash rows on `cols` after normalising dtypes, so int/float and string
    representations of the same value hash identically."""
    norm = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            norm[c] = s.astype(float)
        else:
            num = pd.to_numeric(s, errors="coerce")
            norm[c] = num if num.notna().sum() == s.notna().sum() else s.astype("string")
    return pd.util.hash_pandas_object(pd.DataFrame(norm, index=df.index), index=False)


def check_overlap(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
                  ids: list[str], max_frac: float) -> list[Finding]:
    out = []
    common = [c for c in feats if c in test.columns and c not in ids]
    if not common:
        return out
    tr_h = set(row_hashes(train, common))
    te_h = row_hashes(test, common)
    n = int(te_h.isin(tr_h).sum())
    if n:
        frac = n / len(test)
        sev = "FAIL" if frac > max_frac else "WARN"
        out.append(Finding("row_overlap", sev, "train/test",
                           f"{n} test row(s) ({frac:.2%}) also appear in train "
                           f"(compared on {len(common)} feature columns, id columns excluded)"))
    for c in ids:
        if c in train.columns and c in test.columns:
            shared = len(set(train[c].dropna()) & set(test[c].dropna()))
            if shared:
                out.append(Finding("row_overlap", "WARN", "train/test",
                                   f"{shared} '{c}' value(s) appear in both train and test"))
    return out


# --------------------------------------------------------------------------- #
# Fold driver
# --------------------------------------------------------------------------- #

def audit_fold(dataset: str, fold: str, train_path: Path, test_path: Path,
               labels_path: Optional[Path], args) -> tuple[FoldReport, dict]:
    train, test = read_csv(train_path), read_csv(test_path)
    labels = read_csv(labels_path) if labels_path else None
    extra_ids = set(args.id_cols or [])

    target, how = resolve_target(train, test, labels, args.target, extra_ids)
    rep = FoldReport(dataset, fold, str(train_path), str(test_path),
                     str(labels_path) if labels_path else None, target, None,
                     len(train), len(test), train.shape[1], test.shape[1])
    rep.notes.append(f"target: {target} ({how})")
    ctx = {"columns": list(train.columns), "test_columns": list(test.columns), "test_hashes": None}
    if target is None:
        rep.findings.append(Finding("target", "WARN", "train", how))
        return rep, ctx

    task = args.task or infer_task(train[target])
    rep.task = task
    ids = id_columns(train, extra_ids)
    feats = [c for c in train.columns if c != target]
    model_feats = [c for c in feats if c not in ids]

    # 1. target column in the test file
    test_target = ci_lookup(test.columns, target)
    if test_target:
        note = "" if test_target == target else f" (named '{test_target}' in test)"
        rep.findings.append(Finding("target_in_test", "FAIL", "test",
                                    f"target column '{target}' is present in the test file{note}"))

    # held-out labels for the test-side checks
    y_test = None
    if test_target:
        y_test = test[test_target]
    elif labels is not None:
        lab = ci_lookup(labels.columns, target)
        if lab is None:
            non_id = [c for c in labels.columns if c not in id_columns(labels, extra_ids)]
            lab = non_id[0] if len(non_id) == 1 else None
        if lab is not None and len(labels) == len(test):
            y_test = labels[lab]
        else:
            rep.notes.append("labels file found but could not be aligned with the test file")

    # 2 + 3 on train, and on test when labels are available
    sides = [("train", train[model_feats], train[target])]
    if y_test is not None:
        test_feats = [c for c in model_feats if c in test.columns]
        sides.append(("test", test[test_feats], y_test.reset_index(drop=True)))
    else:
        rep.notes.append("no held-out labels found; reconstruction checks ran on train only")

    for split, X, y in sides:
        X = X.reset_index(drop=True)
        rep.findings += check_single_feature(X, y, task, split)
        found, notes = check_additive_pair(X, y, split, seed=args.seed)
        rep.findings += found
        rep.notes += notes

    # 4. schema
    rep.findings += check_schema(feats, test, target)

    # 5. row overlap
    rep.findings += check_overlap(train, test, feats, ids, args.max_overlap_frac)

    common = [c for c in model_feats if c in test.columns]
    if common:
        ctx["test_hashes"] = set(row_hashes(test, common))
        ctx["hash_cols"] = tuple(common)
    return rep, ctx


def cross_fold_checks(reports: list[FoldReport], ctxs: list[dict]) -> list[Finding]:
    """Within one dataset: compare schemas, target names and test-set overlap across folds."""
    out = []
    if len(reports) < 2:
        return out
    shapes = {(r.fold, r.n_train_cols, r.n_test_cols) for r in reports}
    if len({(a, b) for _, a, b in shapes}) > 1:
        desc = ", ".join(f"{f}: {a}/{b}" for f, a, b in sorted(shapes))
        out.append(Finding("schema", "WARN", "cross-fold",
                           f"column counts (train/test) differ across folds: {desc}"))
    col_sets = [frozenset(c["columns"]) for c in ctxs]
    if len(set(col_sets)) > 1:
        base = col_sets[0]
        for r, cs in zip(reports[1:], col_sets[1:]):
            if cs != base:
                diff = sorted(cs ^ base)
                out.append(Finding("schema", "WARN", "cross-fold",
                                   f"{r.fold} column names differ from {reports[0].fold}: {diff[:10]}"))
    targets = {r.target for r in reports if r.target}
    if len(targets) > 1:
        out.append(Finding("schema", "WARN", "cross-fold",
                           f"target column named differently across folds: {sorted(targets)}"))
    # test sets of different folds should not share rows
    for i in range(len(ctxs)):
        for j in range(i + 1, len(ctxs)):
            a, b = ctxs[i], ctxs[j]
            if a["test_hashes"] and b["test_hashes"] and a.get("hash_cols") == b.get("hash_cols"):
                n = len(a["test_hashes"] & b["test_hashes"])
                if n:
                    out.append(Finding("row_overlap", "WARN", "cross-fold",
                                       f"test sets of {reports[i].fold} and {reports[j].fold} "
                                       f"share {n} distinct row(s)"))
    return out


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover(root: Path) -> list[tuple[str, str, Path, Path, Optional[Path]]]:
    """Find (dataset, fold, train, test, labels) tuples under `root`.

    A train file is any CSV whose name contains 'train'; its test file is the
    same name with the last 'train' replaced by 'test'.
    """
    found = []
    for tr in sorted(root.rglob("*.csv")):
        name = tr.name
        if "train" not in name.lower():
            continue
        idx = name.lower().rfind("train")
        te = tr.with_name(name[:idx] + "test" + name[idx + 5:])
        if not te.exists():
            te = tr.with_name(name[:idx] + "TEST" + name[idx + 5:])
            if not te.exists():
                continue
        labels = None
        stem = te.stem
        for cand in [f"{stem}_y.csv", f"{stem}_labels.csv", *LABEL_FILE_NAMES]:
            p = te.with_name(cand)
            if p.exists() and p != te:
                labels = p
                break
        rel = tr.relative_to(root)
        parts = rel.parts
        dataset = parts[0] if len(parts) > 1 else root.name
        fold = "/".join(parts[1:-1]) if len(parts) > 2 else re.sub(r"(?i)_?train", "", tr.stem) or tr.stem
        found.append((dataset, fold or ".", tr, te, labels))
    return found


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_report(by_dataset: dict, cross: dict) -> None:
    for ds, reps in by_dataset.items():
        status = "FAIL" if any(r.status == "FAIL" for r in reps) else (
            "WARN" if any(r.status == "WARN" for r in reps) or cross.get(ds) else "PASS")
        print(f"\n=== {ds}  [{status}]")
        for r in reps:
            print(f"  -- {r.fold}  [{r.status}]  train={r.n_train}x{r.n_train_cols}  "
                  f"test={r.n_test}x{r.n_test_cols}  task={r.task}")
            for n in r.notes:
                print(f"       . {n}")
            for f in r.findings:
                r2 = f"  (R2={f.r2:.6f})" if f.r2 is not None else ""
                print(f"       {f.severity}  {f.check:<15} [{f.split}] {f.detail}{r2}")
        for f in cross.get(ds, []):
            print(f"     {f.severity}  {f.check:<15} [{f.split}] {f.detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="directory to search for train/test fold files")
    src.add_argument("--train", type=Path, help="single-fold mode: training CSV")
    ap.add_argument("--test", type=Path, help="single-fold mode: test CSV")
    ap.add_argument("--test-labels", type=Path, help="single-fold mode: held-out labels CSV")
    ap.add_argument("--target", help="target column name (otherwise inferred)")
    ap.add_argument("--task", choices=["classification", "regression"], help="otherwise inferred")
    ap.add_argument("--id-cols", nargs="*", help="extra identifier columns to ignore")
    ap.add_argument("--max-overlap-frac", type=float, default=0.0,
                    help="row-overlap fraction above which a fold fails (default 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, help="write the full report as JSON")
    args = ap.parse_args()

    if args.train:
        if not args.test:
            ap.error("--train requires --test")
        p = args.train.resolve().parent
        jobs = [(p.parent.name or "dataset", p.name or args.train.stem,
                 args.train, args.test, args.test_labels)]
    else:
        jobs = discover(args.root)
        if not jobs:
            print(f"No train/test CSV pairs found under {args.root}", file=sys.stderr)
            return 2

    by_dataset: dict[str, list[FoldReport]] = {}
    ctx_by_dataset: dict[str, list[dict]] = {}
    for dataset, fold, tr, te, lab in jobs:
        try:
            rep, ctx = audit_fold(dataset, fold, tr, te, lab, args)
        except Exception as e:  # keep auditing the remaining folds
            rep = FoldReport(dataset, fold, str(tr), str(te), str(lab) if lab else None, None, None)
            rep.findings.append(Finding("error", "FAIL", "-", f"{type(e).__name__}: {e}"))
            ctx = {"columns": [], "test_columns": [], "test_hashes": None}
        by_dataset.setdefault(dataset, []).append(rep)
        ctx_by_dataset.setdefault(dataset, []).append(ctx)

    cross = {ds: cross_fold_checks(reps, ctx_by_dataset[ds]) for ds, reps in by_dataset.items()}
    print_report(by_dataset, cross)

    n_fail = sum(r.status == "FAIL" for reps in by_dataset.values() for r in reps)
    n_folds = sum(len(r) for r in by_dataset.values())
    failed_ds = [ds for ds, reps in by_dataset.items() if any(r.status == "FAIL" for r in reps)]
    print(f"\n{n_folds} fold(s) in {len(by_dataset)} dataset(s); "
          f"{n_fail} fold(s) failed; datasets with failures: {failed_ds or 'none'}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            ds: {
                "status": "FAIL" if ds in failed_ds else "PASS",
                "folds": [{**{k: v for k, v in asdict(r).items() if k != "findings"},
                           "status": r.status,
                           "findings": [asdict(f) for f in r.findings]} for r in reps],
                "cross_fold": [asdict(f) for f in cross[ds]],
            }
            for ds, reps in by_dataset.items()
        }
        args.out_json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"report written to {args.out_json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
