"""Choose one configuration on validation, then evaluate its frozen snapshot."""

from dataclasses import asdict, dataclass
import json

import pandas as pd

from training.data_splitter import DataSplitter, WindowConfig
from training.evaluation import evaluate_predictions
from training.models import create_model
from training.walk_forward import WalkForwardRunner


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    parameters: dict
    window: WindowConfig

    @classmethod
    def from_dict(cls, specification):
        return cls(specification["name"], specification["model"],
                   dict(specification.get("parameters", {})),
                   WindowConfig(**specification["window"]))

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Candidate names must be non-empty")
        create_model(self.model, self.parameters)  # Validate before fitting.


@dataclass(frozen=True)
class FrozenSelection:
    """Immutable JSON snapshot: subsequent edits to candidates cannot retune it."""
    specification_json: str

    def to_dict(self):
        return json.loads(self.specification_json)

    @classmethod
    def from_dict(cls, specification):
        if specification.get("validation_start") != "2018-01-01" or specification.get("validation_end_exclusive") != "2022-01-01":
            raise ValueError("Selection must use the declared 2018--2021 validation period")
        if specification.get("selection_metric") not in ("mae", "rmse", "directional_accuracy"):
            raise ValueError("Unknown selection metric")
        Candidate.from_dict(specification["candidate"])
        return cls(json.dumps(specification, sort_keys=True, allow_nan=False))


@dataclass
class SelectionResult:
    selected: FrozenSelection
    leaderboard: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    metrics: dict[str, pd.DataFrame]


def select_configuration(model_ready, candidates, metric="mae"):
    """Pool validation errors; ties break alphabetically by candidate name.

    Test rows and outcomes are removed before any candidate run. Built-in
    candidates are tabular; custom sequence models use WalkForwardRunner.
    """
    if metric not in ("mae", "rmse", "directional_accuracy"):
        raise ValueError("Selection metric must be mae, rmse, or directional_accuracy")
    candidates = list(candidates)
    if not candidates or len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("Provide candidates with unique names")
    # Snapshot candidates before running; parameters supplied by callers may be mutable.
    candidates = [Candidate.from_dict(asdict(candidate)) for candidate in candidates]
    dates = pd.to_datetime(model_ready.date, utc=True, errors="raise")
    before_test = model_ready.loc[dates < DataSplitter.TEST_START].copy()
    exits = pd.to_datetime(before_test.target_exit_date, utc=True, errors="raise")
    validation_input = before_test.loc[exits < DataSplitter.TEST_START]
    rows, predictions, metrics, reference_dates = [], {}, {}, None
    for candidate in candidates:
        run = WalkForwardRunner().run(
            validation_input, candidate.window,
            model_factory=lambda c=candidate: create_model(c.model, c.parameters),
        )
        keys = run.predictions[["asset", "date"]].reset_index(drop=True)
        if reference_dates is not None and not keys.equals(reference_dates):
            raise ValueError("Candidates must be compared on identical prediction dates")
        reference_dates = keys
        report = evaluate_predictions(run.predictions)
        overall = report.loc[(report.scope == "overall") & (report.predictor == "model")].iloc[0]
        rows.append({"candidate": candidate.name, "model": candidate.model,
                     **{key: overall[key] for key in ("n_samples", "mae", "rmse", "directional_accuracy")}})
        predictions[candidate.name] = run.predictions
        metrics[candidate.name] = report
    leaderboard = pd.DataFrame(rows).sort_values(
        [metric, "candidate"], ascending=[metric != "directional_accuracy", True], kind="stable"
    ).reset_index(drop=True)
    winner = next(candidate for candidate in candidates if candidate.name == leaderboard.candidate.iloc[0])
    frozen = FrozenSelection.from_dict({
        "candidate": asdict(winner), "selection_metric": metric,
        "validation_score": float(leaderboard[metric].iloc[0]),
        "validation_start": "2018-01-01", "validation_end_exclusive": "2022-01-01",
        "tie_break": "candidate name ascending", "dataset_dimensions": "tabular",
    })
    return SelectionResult(frozen, leaderboard, predictions, metrics)


def evaluate_frozen_configuration(model_ready, selected: FrozenSelection):
    """Use frozen settings; retrain on known history at each test cutoff.

    Earlier test observations may enter later training windows once their
    labels are observable. No settings are selected using test outcomes.
    """
    specification = selected.to_dict()
    candidate = Candidate.from_dict(specification["candidate"])
    result = WalkForwardRunner().run(
        model_ready, candidate.window,
        model_factory=lambda: create_model(candidate.model, candidate.parameters),
        period="test",
    )
    return result, evaluate_predictions(result.predictions)
