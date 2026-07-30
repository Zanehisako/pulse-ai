"""
drift_detection.py
==================
Pure statistical drift-detection utilities.

No Django, no MLflow — those integrations live in the callers
(drift_monitor.py for MLflow, Django views for the REST API).

Supported methods
-----------------
* **KS test** (Kolmogorov–Smirnov) — two-sample test for numerical features.
* **PSI** (Population Stability Index) — binned distribution comparison.
* **Chi-squared test** — categorical feature drift.
* **Jensen–Shannon divergence** — symmetric, bounded divergence measure.
* **Prediction drift** — distribution shift on model outputs.

All public helpers return :class:`FeatureDriftResult` or
:class:`DriftReport` dataclasses that serialise cleanly to JSON via
``.to_dict()``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger("pios.drift_detection")

# ── Thresholds ─────────────────────────────────────────────────────────
DEFAULT_KS_PVALUE_THRESHOLD = 0.05  # reject H₀ (same dist) below this
DEFAULT_PSI_THRESHOLD = 0.2  # PSI > 0.2 → significant drift
DEFAULT_PSI_WARNING = 0.1  # 0.1–0.2 → moderate drift
DEFAULT_CHI2_PVALUE_THRESHOLD = 0.05
DEFAULT_JS_THRESHOLD = 0.1

# Tiny floor used when computing log-ratios so we never hit log(0).
_EPS = 1e-8

# Minimum sample size we consider meaningful for any statistical test.
_MIN_SAMPLE_SIZE = 2


# ── Data classes ───────────────────────────────────────────────────────


@dataclass
class FeatureDriftResult:
    """Drift result for a single feature."""

    feature_name: str
    dtype: str  # "numerical" or "categorical"
    test_name: str  # "ks", "psi", "chi2", "js"
    statistic: float
    p_value: float | None = None  # None for PSI / JS (no p-value)
    threshold: float = 0.0
    drift_detected: bool = False
    severity: str = "none"  # "none", "warning", "critical"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a plain ``dict`` (JSON-safe)."""
        return asdict(self)


@dataclass
class DriftReport:
    """Aggregated drift report for one model."""

    model_id: str
    reference_size: int
    current_size: int
    features_checked: int
    features_drifted: int
    overall_drift_detected: bool
    overall_drift_score: float  # fraction of features drifted
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    prediction_drift: FeatureDriftResult | None = None
    summary: str = ""

    def to_dict(self) -> dict:
        """Serialise the full report to a plain ``dict``."""
        return asdict(self)


# ── Helpers (private) ──────────────────────────────────────────────────


def _severity_from_pvalue(p_value: float, threshold: float) -> str:
    """Map a *p*-value to a human-readable severity label.

    * ``"critical"`` — *p* < *threshold*
    * ``"warning"``  — *p* < 2 × *threshold* (borderline)
    * ``"none"``     — otherwise
    """
    if p_value < threshold:
        return "critical"
    if p_value < threshold * 2:
        return "warning"
    return "none"


def _severity_from_score(
    score: float,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    """Map a non-negative score (PSI, JS, …) to a severity label."""
    if score >= critical_threshold:
        return "critical"
    if score >= warning_threshold:
        return "warning"
    return "none"


def _validate_arrays(
    reference: np.ndarray,
    current: np.ndarray,
    name: str = "feature",
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sanitise & validate input arrays.

    Returns
    -------
    ref_clean, cur_clean, is_valid
        ``is_valid`` is ``False`` when the inputs are too small or
        entirely NaN, in which case the caller should return a
        "skip" result.
    """
    try:
        ref = np.asarray(reference, dtype=float).ravel()
        cur = np.asarray(current, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Cannot convert '%s' to numeric array — skipping (%s).",
            name,
            exc,
        )
        return np.array([]), np.array([]), False

    # Drop NaN / Inf
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) < _MIN_SAMPLE_SIZE or len(cur) < _MIN_SAMPLE_SIZE:
        logger.warning(
            "Skipping drift check for '%s': insufficient data (ref=%d, cur=%d).",
            name,
            len(ref),
            len(cur),
        )
        return ref, cur, False

    return ref, cur, True


def _make_skip_result(
    feature_name: str,
    dtype: str,
    test_name: str,
    reason: str,
) -> FeatureDriftResult:
    """Return a neutral ``FeatureDriftResult`` when we cannot run a test."""
    return FeatureDriftResult(
        feature_name=feature_name,
        dtype=dtype,
        test_name=test_name,
        statistic=0.0,
        p_value=None,
        threshold=0.0,
        drift_detected=False,
        severity="none",
        details={"skipped": True, "reason": reason},
    )


# ── KS test ───────────────────────────────────────────────────────────


def compute_ks_test(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = DEFAULT_KS_PVALUE_THRESHOLD,
    feature_name: str = "unknown",
) -> FeatureDriftResult:
    """Two-sample Kolmogorov–Smirnov test for a numerical feature.

    Parameters
    ----------
    reference:
        1-D array of reference (training / baseline) values.
    current:
        1-D array of current (production / serving) values.
    threshold:
        *p*-value below which we declare drift.
    feature_name:
        Human-readable label used in the result.

    Returns
    -------
    FeatureDriftResult
        ``test_name="ks"``, ``dtype="numerical"``.
    """
    ref, cur, valid = _validate_arrays(reference, current, feature_name)
    if not valid:
        return _make_skip_result(feature_name, "numerical", "ks", "insufficient data")

    ks_result = stats.ks_2samp(ref, cur)
    ks_stat = float(ks_result[0])  # type: ignore[arg-type]  # statistic
    p_value = float(ks_result[1])  # type: ignore[arg-type]  # pvalue
    severity = _severity_from_pvalue(p_value, threshold)

    return FeatureDriftResult(
        feature_name=feature_name,
        dtype="numerical",
        test_name="ks",
        statistic=ks_stat,
        p_value=p_value,
        threshold=threshold,
        drift_detected=p_value < threshold,
        severity=severity,
        details={
            "reference_mean": float(np.mean(ref)),
            "current_mean": float(np.mean(cur)),
            "reference_std": float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0,
            "current_std": float(np.std(cur, ddof=1)) if len(cur) > 1 else 0.0,
            "reference_size": len(ref),
            "current_size": len(cur),
        },
    )


# ── PSI ────────────────────────────────────────────────────────────────


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    threshold: float = DEFAULT_PSI_THRESHOLD,
    feature_name: str = "unknown",
) -> FeatureDriftResult:
    """Population Stability Index for a numerical feature.

    The reference distribution is used to define equal-width bins, and
    both distributions are binned accordingly.  Small bin proportions
    are clipped to :data:`_EPS` to avoid ``log(0)``.

    Parameters
    ----------
    reference, current:
        1-D numeric arrays.
    n_bins:
        Number of equal-width bins derived from the reference range.
    threshold:
        PSI value above which drift is flagged as **critical**.
    feature_name:
        Label for the result.

    Returns
    -------
    FeatureDriftResult
        ``test_name="psi"``, ``p_value=None`` (PSI is not a test statistic
        with a *p*-value).
    """
    ref, cur, valid = _validate_arrays(reference, current, feature_name)
    if not valid:
        return _make_skip_result(feature_name, "numerical", "psi", "insufficient data")

    # Constant feature → no drift measurable.
    ref_min, ref_max = float(np.min(ref)), float(np.max(ref))
    if ref_min == ref_max:
        return _make_skip_result(
            feature_name, "numerical", "psi", "constant reference feature"
        )

    # Build bins from the reference range.
    bin_edges = np.linspace(ref_min, ref_max, n_bins + 1)
    # Extend outer edges slightly so all values fall inside.
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=bin_edges)[0].astype(float)

    # Convert to proportions and clip.
    ref_pct = np.clip(ref_counts / ref_counts.sum(), _EPS, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), _EPS, None)

    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    severity = _severity_from_score(psi_value, DEFAULT_PSI_WARNING, threshold)

    return FeatureDriftResult(
        feature_name=feature_name,
        dtype="numerical",
        test_name="psi",
        statistic=float(psi_value),
        p_value=None,
        threshold=threshold,
        drift_detected=psi_value >= threshold,
        severity=severity,
        details={
            "n_bins": n_bins,
            "bin_psi_values": (
                ((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).tolist()
            ),
            "reference_size": len(ref),
            "current_size": len(cur),
        },
    )


# ── Chi-squared test ──────────────────────────────────────────────────


def compute_chi2_test(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = DEFAULT_CHI2_PVALUE_THRESHOLD,
    feature_name: str | None = None,
) -> FeatureDriftResult:
    """Chi-squared goodness-of-fit test for a categorical feature.

    Categories are aligned across both series — categories present in
    only one series receive a count of zero in the other.

    Parameters
    ----------
    reference, current:
        Pandas Series of categorical values.
    threshold:
        *p*-value below which drift is declared.
    feature_name:
        If ``None`` the Series ``.name`` attribute is used (falling
        back to ``"unknown"``).

    Returns
    -------
    FeatureDriftResult
        ``test_name="chi2"``, ``dtype="categorical"``.
    """
    fname = feature_name or getattr(reference, "name", None) or "unknown"

    ref_series = pd.Series(reference).dropna()
    cur_series = pd.Series(current).dropna()

    if len(ref_series) < _MIN_SAMPLE_SIZE or len(cur_series) < _MIN_SAMPLE_SIZE:
        return _make_skip_result(fname, "categorical", "chi2", "insufficient data")

    ref_counts = ref_series.value_counts()
    cur_counts = cur_series.value_counts()

    # Align on the union of categories.
    all_categories = sorted(
        set(ref_counts.index.tolist()) | set(cur_counts.index.tolist()),
        key=str,
    )
    ref_aligned = np.array([ref_counts.get(c, 0) for c in all_categories], dtype=float)
    cur_aligned = np.array([cur_counts.get(c, 0) for c in all_categories], dtype=float)

    # If all counts are zero on either side we cannot run the test.
    if ref_aligned.sum() == 0 or cur_aligned.sum() == 0:
        return _make_skip_result(fname, "categorical", "chi2", "zero total counts")

    # Scale reference counts to match current sample size so we can
    # treat them as "expected" frequencies.
    expected = ref_aligned * (cur_aligned.sum() / ref_aligned.sum())
    # Replace zeros with a tiny value to avoid division errors.
    expected = np.where(expected == 0, _EPS, expected)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        chi2_result = stats.chisquare(cur_aligned, f_exp=expected)
        chi2_stat = float(chi2_result[0])  # type: ignore[arg-type]  # statistic
        p_value = float(chi2_result[1])  # type: ignore[arg-type]  # pvalue

    severity = _severity_from_pvalue(p_value, threshold)

    return FeatureDriftResult(
        feature_name=fname,
        dtype="categorical",
        test_name="chi2",
        statistic=float(chi2_stat),
        p_value=float(p_value),
        threshold=threshold,
        drift_detected=p_value < threshold,
        severity=severity,
        details={
            "n_categories": len(all_categories),
            "categories": [str(c) for c in all_categories],
            "reference_distribution": (ref_aligned / ref_aligned.sum()).tolist(),
            "current_distribution": (cur_aligned / cur_aligned.sum()).tolist(),
            "reference_size": int(ref_aligned.sum()),
            "current_size": int(cur_aligned.sum()),
        },
    )


# ── Jensen–Shannon divergence ─────────────────────────────────────────


def compute_js_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 50,
    threshold: float = DEFAULT_JS_THRESHOLD,
    feature_name: str = "unknown",
) -> FeatureDriftResult:
    """Jensen–Shannon divergence between two numerical distributions.

    The distributions are discretised into *n_bins* equal-width bins
    (range taken from the union of both arrays).  The square of
    ``scipy.spatial.distance.jensenshannon`` gives the JS *divergence*
    (bounded 0–1 when using base-2 log, 0–ln2 for natural log; scipy
    returns the **distance**, i.e. the square root of the divergence).

    We report the **divergence** (squared distance) as the statistic.

    Parameters
    ----------
    reference, current:
        1-D numeric arrays.
    n_bins:
        Number of histogram bins.
    threshold:
        JS divergence above which drift is declared.
    feature_name:
        Label for the result.

    Returns
    -------
    FeatureDriftResult
        ``test_name="js"``, ``p_value=None``.
    """
    ref, cur, valid = _validate_arrays(reference, current, feature_name)
    if not valid:
        return _make_skip_result(feature_name, "numerical", "js", "insufficient data")

    # Determine common bin edges from the union of both arrays.
    combined_min = min(float(np.min(ref)), float(np.min(cur)))
    combined_max = max(float(np.max(ref)), float(np.max(cur)))
    if combined_min == combined_max:
        return _make_skip_result(
            feature_name, "numerical", "js", "constant values in both arrays"
        )

    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)
    # Extend edges so nothing falls outside.
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_hist = np.histogram(ref, bins=bin_edges)[0].astype(float)
    cur_hist = np.histogram(cur, bins=bin_edges)[0].astype(float)

    # Normalise to probability distributions (add eps to avoid zeros).
    ref_prob = (ref_hist + _EPS) / (ref_hist + _EPS).sum()
    cur_prob = (cur_hist + _EPS) / (cur_hist + _EPS).sum()

    js_distance = jensenshannon(ref_prob, cur_prob, base=2.0)
    js_divergence = float(js_distance**2)  # divergence = distance²

    severity = _severity_from_score(js_divergence, threshold * 0.5, threshold)

    return FeatureDriftResult(
        feature_name=feature_name,
        dtype="numerical",
        test_name="js",
        statistic=js_divergence,
        p_value=None,
        threshold=threshold,
        drift_detected=js_divergence >= threshold,
        severity=severity,
        details={
            "js_distance": float(js_distance),
            "n_bins": n_bins,
            "reference_size": len(ref),
            "current_size": len(cur),
        },
    )


# ── Feature drift (batch) ─────────────────────────────────────────────


def _infer_feature_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Heuristically split columns into numerical vs categorical.

    Columns are **skipped** (neither list) when they are:

    * datetime / timedelta / period types
    * entity-id columns (suffix ``_id`` with high cardinality)

    A column is treated as *categorical* when:

    * its pandas dtype is ``object``, ``category``, or ``bool``; **or**
    * it has ≤ 20 unique values **and** is integer-typed.

    Everything else is treated as numerical.
    """
    numerical: list[str] = []
    categorical: list[str] = []
    for col in df.columns:
        dtype = df[col].dtype

        # ── Skip datetime / timedelta / period columns ────────────
        if pd.api.types.is_datetime64_any_dtype(dtype):
            logger.debug("Skipping datetime column '%s'", col)
            continue
        if pd.api.types.is_timedelta64_dtype(dtype):
            logger.debug("Skipping timedelta column '%s'", col)
            continue
        if isinstance(dtype, pd.PeriodDtype):
            logger.debug("Skipping period column '%s'", col)
            continue

        # ── Skip high-cardinality ID columns ──────────────────────
        if col.endswith("_id") and dtype.kind in ("O", "S", "U"):
            logger.debug("Skipping entity-id column '%s'", col)
            continue

        # ── Categorical vs numerical ─────────────────────────────
        if dtype.kind in ("O", "S", "U") or pd.api.types.is_categorical_dtype(dtype):
            categorical.append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            categorical.append(col)
        elif pd.api.types.is_integer_dtype(dtype) and df[col].nunique() <= 20:
            categorical.append(col)
        else:
            numerical.append(col)
    return numerical, categorical


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    config: dict | None = None,
) -> list[FeatureDriftResult]:
    """Run drift tests on every feature present in both dataframes.

    Parameters
    ----------
    reference_df, current_df:
        DataFrames sharing (a subset of) the same columns.
    numerical_features:
        Explicit list of numerical columns.  Auto-detected when
        ``None``.
    categorical_features:
        Explicit list of categorical columns.  Auto-detected when
        ``None``.
    config:
        Optional per-feature or global threshold overrides::

            {
                "ks_threshold": 0.05,
                "psi_threshold": 0.2,
                "chi2_threshold": 0.05,
                "feature_overrides": {
                    "age": {"ks_threshold": 0.01},
                },
            }

    Returns
    -------
    list[FeatureDriftResult]
        One result per feature (numerical features yield *both* a KS
        **and** a PSI result).
    """
    config = config or {}
    results: list[FeatureDriftResult] = []

    # Auto-detect if not supplied.
    if numerical_features is None and categorical_features is None:
        numerical_features, categorical_features = _infer_feature_types(reference_df)
    numerical_features = numerical_features or []
    categorical_features = categorical_features or []

    # Only test features present in both dataframes.
    common_cols = set(reference_df.columns) & set(current_df.columns)

    overrides = config.get("feature_overrides", {})

    # ── Numerical features ─────────────────────────────────────────
    global_ks = config.get("ks_threshold", DEFAULT_KS_PVALUE_THRESHOLD)
    global_psi = config.get("psi_threshold", DEFAULT_PSI_THRESHOLD)

    for feat in numerical_features:
        if feat not in common_cols:
            logger.info("Feature '%s' not in current data — skipping.", feat)
            results.append(
                _make_skip_result(feat, "numerical", "ks", "missing in current data")
            )
            continue

        feat_cfg = overrides.get(feat, {})
        ks_thresh = feat_cfg.get("ks_threshold", global_ks)
        psi_thresh = feat_cfg.get("psi_threshold", global_psi)

        ref_vals: np.ndarray = reference_df[feat].to_numpy()
        cur_vals: np.ndarray = current_df[feat].to_numpy()

        results.append(
            compute_ks_test(ref_vals, cur_vals, ks_thresh, feature_name=feat)
        )
        results.append(
            compute_psi(ref_vals, cur_vals, threshold=psi_thresh, feature_name=feat)
        )

    # ── Categorical features ───────────────────────────────────────
    global_chi2 = config.get("chi2_threshold", DEFAULT_CHI2_PVALUE_THRESHOLD)

    for feat in categorical_features:
        if feat not in common_cols:
            logger.info("Feature '%s' not in current data — skipping.", feat)
            results.append(
                _make_skip_result(
                    feat, "categorical", "chi2", "missing in current data"
                )
            )
            continue

        feat_cfg = overrides.get(feat, {})
        chi2_thresh = feat_cfg.get("chi2_threshold", global_chi2)

        ref_series = pd.Series(reference_df[feat])
        cur_series = pd.Series(current_df[feat])
        results.append(
            compute_chi2_test(
                ref_series,
                cur_series,
                threshold=chi2_thresh,
                feature_name=feat,
            )
        )

    return results


# ── Prediction drift ──────────────────────────────────────────────────


def detect_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    task_type: str = "regression",
    threshold: float | None = None,
) -> FeatureDriftResult:
    """Detect drift in model predictions.

    Parameters
    ----------
    reference_predictions, current_predictions:
        1-D arrays of model outputs.
    task_type:
        ``"regression"`` → KS test on raw values.
        ``"classification"`` → Chi-squared test on predicted classes.
    threshold:
        Custom threshold; defaults to the standard threshold for the
        chosen test.

    Returns
    -------
    FeatureDriftResult
        With ``feature_name="prediction"``.
    """
    if task_type == "classification":
        thresh = threshold if threshold is not None else DEFAULT_CHI2_PVALUE_THRESHOLD
        ref_series = pd.Series(reference_predictions, name="prediction")
        cur_series = pd.Series(current_predictions, name="prediction")
        result = compute_chi2_test(
            ref_series, cur_series, threshold=thresh, feature_name="prediction"
        )
        result.details["task_type"] = "classification"
        return result

    # Default: regression → KS test.
    thresh = threshold if threshold is not None else DEFAULT_KS_PVALUE_THRESHOLD
    result = compute_ks_test(
        np.asarray(reference_predictions),
        np.asarray(current_predictions),
        threshold=thresh,
        feature_name="prediction",
    )
    result.details["task_type"] = "regression"
    return result


# ── Full report builder ───────────────────────────────────────────────


def build_drift_report(
    model_id: str,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reference_predictions: np.ndarray | None = None,
    current_predictions: np.ndarray | None = None,
    config: dict | None = None,
) -> DriftReport:
    """Build a complete :class:`DriftReport` for a model.

    This is the **main entry point** for callers that want a single,
    self-contained drift assessment.

    Parameters
    ----------
    model_id:
        Identifier for the model being monitored.
    reference_df, current_df:
        Feature DataFrames (reference = training / baseline, current =
        production / new batch).
    reference_predictions, current_predictions:
        Optional prediction arrays.  When both are supplied, prediction
        drift is also assessed.
    config:
        Threshold overrides — forwarded to
        :func:`detect_feature_drift`.  May additionally contain:

        * ``task_type`` (``str``): ``"regression"`` or
          ``"classification"`` (default ``"regression"``).

    Returns
    -------
    DriftReport
    """
    config = config or {}

    # ── Feature drift ──────────────────────────────────────────────
    feature_results = detect_feature_drift(
        reference_df,
        current_df,
        numerical_features=config.get("numerical_features"),
        categorical_features=config.get("categorical_features"),
        config=config,
    )

    # Count unique features that drifted (a feature can appear twice
    # if both KS and PSI were run).
    drifted_features: set[str] = set()
    checked_features: set[str] = set()
    for r in feature_results:
        # Skip entries that were skipped themselves.
        if r.details.get("skipped"):
            continue
        checked_features.add(r.feature_name)
        if r.drift_detected:
            drifted_features.add(r.feature_name)

    features_checked = len(checked_features)
    features_drifted = len(drifted_features)
    drift_score = features_drifted / features_checked if features_checked else 0.0

    # ── Prediction drift ───────────────────────────────────────────
    pred_result: FeatureDriftResult | None = None
    if reference_predictions is not None and current_predictions is not None:
        task_type = config.get("task_type", "regression")
        pred_result = detect_prediction_drift(
            reference_predictions,
            current_predictions,
            task_type=task_type,
            threshold=config.get("prediction_threshold"),
        )

    overall_drift = features_drifted > 0 or (
        pred_result is not None and pred_result.drift_detected
    )

    # ── Build human-readable summary ───────────────────────────────
    summary_parts: list[str] = []
    if features_checked == 0:
        summary_parts.append("No features could be checked for drift.")
    elif features_drifted == 0:
        summary_parts.append(f"No drift detected across {features_checked} feature(s).")
    else:
        summary_parts.append(
            f"Drift detected in {features_drifted}/{features_checked} "
            f"feature(s): {sorted(drifted_features)}."
        )
    if pred_result is not None:
        if pred_result.drift_detected:
            summary_parts.append(
                f"Prediction drift detected "
                f"(statistic={pred_result.statistic:.4f}, "
                f"severity={pred_result.severity})."
            )
        else:
            summary_parts.append("No prediction drift detected.")

    return DriftReport(
        model_id=model_id,
        reference_size=len(reference_df),
        current_size=len(current_df),
        features_checked=features_checked,
        features_drifted=features_drifted,
        overall_drift_detected=overall_drift,
        overall_drift_score=drift_score,
        feature_results=feature_results,
        prediction_drift=pred_result,
        summary=" ".join(summary_parts),
    )
