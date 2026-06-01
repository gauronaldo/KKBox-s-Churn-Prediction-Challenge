"""Small FastAPI app for interactive KKBox churn scoring.

The app scores engineered feature rows, not raw KKBox transaction/log tables.
This keeps online inference aligned with the trained preprocessor and avoids
running the heavy ingestion/feature aggregation path inside a web request.
"""

from __future__ import annotations

import io
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocess import apply_preprocessor, load_preprocessor
from src.utils.config import get_path, get_value, load_config


class JsonScoringRequest(BaseModel):
    """JSON payload for scoring one or more engineered feature rows."""

    records: list[dict[str, Any]] = Field(..., min_length=1)


class ScoringArtifacts(BaseModel):
    """Loaded model artifacts used by the scoring app."""

    model_config = {"arbitrary_types_allowed": True}

    model: Any
    preprocessor: Any
    threshold: float
    id_col: str
    target_col: str


app = FastAPI(title="KKBox Churn Scoring", version="1.0.0")

EXAMPLE_RECORD: dict[str, Any] = {
    "msno": "demo_user_001",
    "city": 1,
    "bd": 28,
    "gender": "female",
    "registered_via": 7,
    "trans_count": 4,
    "total_spend": 596.0,
    "mean_spend": 149.0,
    "max_spend": 149.0,
    "cancel_count": 0,
    "cancel_rate": 0.0,
    "auto_renew_rate": 1.0,
    "mean_plan_days": 30.0,
    "mean_plan_price": 149.0,
    "mean_discount_rate": 0.0,
    "days_since_last_transaction": 5,
    "active_days": 18,
    "total_secs": 24500.0,
    "mean_secs": 1361.1,
    "total_25": 42.0,
    "total_50": 18.0,
    "total_75": 12.0,
    "total_985": 8.0,
    "total_100": 320.0,
    "total_unq": 210.0,
    "mean_unq": 11.7,
    "completion_rate": 0.8,
    "mean_completion_rate": 0.78,
    "days_since_last_log": 2,
}

FORM_SECTIONS: list[dict[str, Any]] = [
    {
        "title": "Customer profile",
        "fields": [
            ("msno", "User ID", "text", "demo_user_001", "Customer identifier"),
            ("city", "City code", "number", "1", "Integer category"),
            ("bd", "Age", "number", "28", "Typical range: 7-70"),
            ("gender", "Gender", "select", "female", "male / female / blank"),
            ("registered_via", "Registration channel", "number", "7", "Integer category"),
        ],
    },
    {
        "title": "Subscription behavior",
        "fields": [
            ("trans_count", "Transactions", "number", "4", "Count, >= 0"),
            ("total_spend", "Total spend", "number", "596", "Amount paid, >= 0"),
            ("mean_spend", "Mean spend", "number", "149", "Average amount paid"),
            ("max_spend", "Max spend", "number", "149", "Largest payment"),
            ("auto_renew_rate", "Auto-renew rate", "number", "1", "Ratio, 0 to 1"),
            ("cancel_rate", "Cancel rate", "number", "0", "Ratio, 0 to 1"),
            ("cancel_count", "Cancel count", "number", "0", "Count, >= 0"),
            ("mean_plan_days", "Mean plan days", "number", "30", "Average plan length"),
            ("mean_plan_price", "Mean plan price", "number", "149", "Average list price"),
            ("mean_discount_rate", "Mean discount rate", "number", "0", "Ratio, 0 to 1"),
            ("days_since_last_transaction", "Days since last transaction", "number", "5", "Recency, >= 0"),
        ],
    },
    {
        "title": "Listening behavior",
        "fields": [
            ("active_days", "Active days", "number", "18", "Count, >= 0"),
            ("total_secs", "Total listening seconds", "number", "24500", "Total seconds, >= 0"),
            ("mean_secs", "Mean seconds per active day", "number", "1361.1", "Average listening depth"),
            ("total_25", "25% plays", "number", "42", "Partial plays, >= 0"),
            ("total_50", "50% plays", "number", "18", "Partial plays, >= 0"),
            ("total_75", "75% plays", "number", "12", "Partial plays, >= 0"),
            ("total_985", "98.5% plays", "number", "8", "Near-complete plays"),
            ("total_100", "Complete plays", "number", "320", "Completed tracks"),
            ("total_unq", "Unique tracks", "number", "210", "Unique tracks heard"),
            ("mean_unq", "Mean unique tracks", "number", "11.7", "Average per active day"),
            ("completion_rate", "Completion rate", "number", "0.8", "Ratio, 0 to 1"),
            ("mean_completion_rate", "Mean completion rate", "number", "0.78", "Ratio, 0 to 1"),
            ("days_since_last_log", "Days since last listening log", "number", "2", "Recency, >= 0"),
        ],
    },
]


def _load_pickle(path: Path) -> Any:
    """Load a pickle artifact from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def _load_artifacts() -> ScoringArtifacts:
    """Load and cache the champion model, preprocessor, and threshold."""

    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    models_dir = get_path(config, "models_dir", base_dir=PROJECT_ROOT)
    model_path = models_dir / str(get_value(config, "artifacts", "champion_model_file"))
    preprocessor_path = models_dir / str(get_value(config, "artifacts", "preprocessor_file"))
    threshold_path = models_dir / str(
        get_value(config, "artifacts", "champion_threshold_file", default="champion_threshold.txt")
    )

    threshold = float(get_value(config, "modeling", "decision_threshold", default=0.5))
    if threshold_path.exists():
        threshold = float(threshold_path.read_text(encoding="utf-8").strip())

    return ScoringArtifacts(
        model=_load_pickle(model_path),
        preprocessor=load_preprocessor(preprocessor_path),
        threshold=threshold,
        id_col=str(get_value(config, "project", "id_col", default="msno")),
        target_col=str(get_value(config, "project", "target_col", default="is_churn")),
    )


def _score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Score engineered feature rows and return a compact prediction table."""

    if frame.empty:
        raise ValueError("Input frame is empty.")

    artifacts = _load_artifacts()
    features = frame.drop(columns=[artifacts.target_col], errors="ignore")
    for column in _required_input_columns(artifacts.preprocessor):
        if column not in features.columns:
            features[column] = pd.NA
    transformed = apply_preprocessor(artifacts.preprocessor, features, split_name="app")
    probabilities = artifacts.model.predict_proba(transformed)[:, 1]

    output = pd.DataFrame(index=frame.index)
    if artifacts.id_col in frame.columns:
        output[artifacts.id_col] = frame[artifacts.id_col].astype("string")
    output["churn_probability"] = probabilities
    output["predicted_churn"] = (probabilities >= artifacts.threshold).astype("int8")
    output["threshold"] = artifacts.threshold
    return output


def _example_csv() -> str:
    """Return a one-row CSV example for the browser UI."""

    return pd.DataFrame([EXAMPLE_RECORD]).to_csv(index=False).strip()


def _form_sections_html() -> str:
    """Render grouped form controls for the browser UI."""

    sections: list[str] = []
    for section in FORM_SECTIONS:
        controls: list[str] = []
        for name, label, input_type, placeholder, help_text in section["fields"]:
            if input_type == "select":
                control = f"""
                  <select id="{name}" data-field="{name}">
                    <option value="">unknown</option>
                    <option value="female">female</option>
                    <option value="male">male</option>
                  </select>
                """
            else:
                step = ' step="any"' if input_type == "number" else ""
                min_value = ' min="0"' if input_type == "number" else ""
                control = (
                    f'<input id="{name}" data-field="{name}" type="{input_type}"'
                    f'{step}{min_value} placeholder="{placeholder}" />'
                )
            controls.append(
                f"""
                <label class="field">
                  <span>{label}</span>
                  {control}
                  <small>{help_text}</small>
                </label>
                """
            )
        sections.append(
            f"""
            <section class="panel">
              <h2>{section["title"]}</h2>
              <div class="fields">
                {''.join(controls)}
              </div>
            </section>
            """
        )
    return "\n".join(sections)


def _required_input_columns(preprocessor: Any) -> list[str]:
    """Return raw feature columns required by the fitted ColumnTransformer."""

    required: list[str] = []
    for _, transformer, columns in getattr(preprocessor, "transformers_", []):
        if transformer == "drop" or columns is None:
            continue
        required.extend([str(column) for column in columns])
    return required


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Render a minimal browser UI for CSV scoring."""

    example_csv = _example_csv()
    form_sections = _form_sections_html()
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>KKBox Churn Scoring</title>
        <style>
          :root {{ color-scheme: light; }}
          body {{ font-family: Arial, sans-serif; margin: 0; color: #17202a; background: #f7f9fb; }}
          main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
          header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-end; margin-bottom: 18px; }}
          h1 {{ margin: 0 0 6px; font-size: 30px; }}
          h2 {{ margin: 0 0 12px; font-size: 18px; }}
          p {{ margin: 0; line-height: 1.45; }}
          textarea {{ width: 100%; min-height: 180px; font-family: Consolas, monospace; font-size: 13px; }}
          button {{ border: 0; padding: 10px 14px; cursor: pointer; background: #155eef; color: white; font-weight: 700; }}
          button.secondary {{ background: #e8eef7; color: #233040; }}
          button:disabled {{ opacity: 0.55; cursor: wait; }}
          input, select, textarea {{ border: 1px solid #ccd6e0; padding: 9px; background: white; }}
          pre {{ background: #111827; color: #e5e7eb; padding: 16px; overflow: auto; }}
          table {{ width: 100%; border-collapse: collapse; }}
          th, td {{ border-bottom: 1px solid #d6dbdf; padding: 8px; text-align: left; }}
          code {{ background: #eef2f3; padding: 1px 4px; }}
          .tabs {{ display: flex; gap: 8px; margin: 18px 0; }}
          .tab {{ background: #e8eef7; color: #233040; }}
          .tab.active {{ background: #155eef; color: white; }}
          .view {{ display: none; }}
          .view.active {{ display: block; }}
          .layout {{ display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(320px, 0.9fr); gap: 18px; }}
          .panel {{ background: white; border: 1px solid #dde5ed; padding: 16px; margin-bottom: 14px; }}
          .fields {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
          .field {{ display: grid; gap: 5px; }}
          .field span {{ font-weight: 700; font-size: 13px; }}
          .field small {{ color: #5f6f82; min-height: 16px; }}
          .actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 14px 0; }}
          .note {{ background: #fff8e1; border-left: 4px solid #f4b400; padding: 10px 12px; margin: 10px 0 16px; }}
          .result-card {{ background: white; border: 1px solid #dde5ed; padding: 18px; position: sticky; top: 18px; }}
          .prob {{ font-size: 42px; font-weight: 800; margin: 8px 0; }}
          .badge {{ display: inline-block; padding: 6px 10px; font-weight: 800; }}
          .badge.high {{ background: #ffe3e3; color: #b42318; }}
          .badge.low {{ background: #e7f8ef; color: #067647; }}
          .muted {{ color: #5f6f82; font-size: 13px; }}
          .advanced {{ margin-top: 18px; }}
          @media (max-width: 920px) {{
            header {{ display: block; }}
            .layout {{ grid-template-columns: 1fr; }}
            .fields {{ grid-template-columns: 1fr; }}
            .result-card {{ position: static; }}
          }}
        </style>
      </head>
      <body>
        <main>
          <header>
            <div>
              <h1>KKBox Churn Scoring</h1>
              <p>Enter a customer profile, predict churn risk, and review the decision threshold.</p>
            </div>
            <div class="muted">Champion threshold is loaded from <code>models/champion_threshold.txt</code>.</div>
          </header>
          <div class="note">
            The form uses the most common engineered user-level features. Missing model columns are filled as missing values for demo use; full production scoring should use the complete feature schema.
          </div>

          <div class="tabs">
            <button id="tabForm" class="tab active" onclick="showView('form')">Single customer</button>
            <button id="tabCsv" class="tab" onclick="showView('csv')">CSV batch</button>
          </div>

          <div id="formView" class="view active">
            <div class="layout">
              <div>
                {form_sections}
                <div class="actions">
                  <button onclick="fillFormExample()">Fill example</button>
                  <button class="secondary" onclick="clearForm()">Clear</button>
                  <button id="predictFormBtn" onclick="scoreForm()">Predict churn</button>
                </div>
              </div>
              <aside class="result-card">
                <h2>Prediction</h2>
                <div id="emptyState" class="muted">No prediction yet.</div>
                <div id="singleResult" style="display:none;">
                  <div class="muted" id="resultUser"></div>
                  <div class="prob" id="resultProb"></div>
                  <div id="resultBadge" class="badge"></div>
                  <table style="margin-top:14px;">
                    <tr><th>Threshold</th><td id="resultThreshold"></td></tr>
                    <tr><th>Decision</th><td id="resultDecision"></td></tr>
                  </table>
                </div>
                <pre id="resultRaw" style="display:none;"></pre>
              </aside>
            </div>
          </div>

          <div id="csvView" class="view advanced">
            <section class="panel">
              <h2>Batch CSV scoring</h2>
              <p class="muted">Paste CSV with a header row or upload a CSV file. One row equals one customer.</p>
              <div class="actions">
                <input id="file" type="file" accept=".csv,text/csv" />
                <button class="secondary" onclick="loadFile()">Load file</button>
                <button class="secondary" onclick="fillCsvExample()">Fill example CSV</button>
                <button onclick="scoreCsv()">Predict batch</button>
              </div>
              <textarea id="csv" placeholder="Paste CSV here, including a header row."></textarea>
              <h2>Batch predictions</h2>
              <div id="batchSummary" class="muted">No batch prediction yet.</div>
              <div id="batchTable"></div>
              <h2>CSV example</h2>
              <pre>{example_csv}</pre>
            </section>
          </div>
        </main>
        <script>
          const exampleCsv = `{example_csv}`;
          const exampleRecord = {EXAMPLE_RECORD};
          function showView(name) {{
            document.getElementById("formView").classList.toggle("active", name === "form");
            document.getElementById("csvView").classList.toggle("active", name === "csv");
            document.getElementById("tabForm").classList.toggle("active", name === "form");
            document.getElementById("tabCsv").classList.toggle("active", name === "csv");
          }}
          async function loadFile() {{
            const file = document.getElementById("file").files[0];
            if (!file) return;
            document.getElementById("csv").value = await file.text();
          }}
          function fillCsvExample() {{
            document.getElementById("csv").value = exampleCsv;
          }}
          function fillFormExample() {{
            for (const [key, value] of Object.entries(exampleRecord)) {{
              const el = document.querySelector(`[data-field="${{key}}"]`);
              if (el) el.value = value;
            }}
          }}
          function clearForm() {{
            document.querySelectorAll("[data-field]").forEach(el => el.value = "");
          }}
          function collectFormRecord() {{
            const record = {{}};
            document.querySelectorAll("[data-field]").forEach(el => {{
              const key = el.dataset.field;
              if (el.value === "") return;
              if (el.type === "number") {{
                const value = Number(el.value);
                if (!Number.isNaN(value)) record[key] = value;
              }} else {{
                record[key] = el.value;
              }}
            }});
            return record;
          }}
          async function scoreForm() {{
            const button = document.getElementById("predictFormBtn");
            button.disabled = true;
            try {{
              const res = await fetch("/predict", {{
                method: "POST",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify({{records: [collectFormRecord()]}})
              }});
              const payload = await res.json();
              if (!res.ok) throw new Error(payload.detail || "Prediction failed");
              renderSingleResult(payload.predictions[0]);
            }} catch (error) {{
              renderError(error.message);
            }} finally {{
              button.disabled = false;
            }}
          }}
          function renderSingleResult(row) {{
            const probability = Number(row.churn_probability);
            const pct = `${{(probability * 100).toFixed(1)}}%`;
            const isChurn = Number(row.predicted_churn) === 1;
            document.getElementById("emptyState").style.display = "none";
            document.getElementById("singleResult").style.display = "block";
            document.getElementById("resultRaw").style.display = "none";
            document.getElementById("resultUser").textContent = row.msno ? `User: ${{row.msno}}` : "Single customer";
            document.getElementById("resultProb").textContent = pct;
            document.getElementById("resultBadge").textContent = isChurn ? "High churn risk" : "Lower churn risk";
            document.getElementById("resultBadge").className = `badge ${{isChurn ? "high" : "low"}}`;
            document.getElementById("resultThreshold").textContent = Number(row.threshold).toFixed(3);
            document.getElementById("resultDecision").textContent = isChurn ? "Target for retention" : "Do not prioritize";
          }}
          function renderError(message) {{
            document.getElementById("emptyState").style.display = "none";
            document.getElementById("singleResult").style.display = "none";
            document.getElementById("resultRaw").style.display = "block";
            document.getElementById("resultRaw").textContent = message;
          }}
          async function scoreCsv() {{
            const body = document.getElementById("csv").value;
            const res = await fetch("/predict-csv", {{
              method: "POST",
              headers: {{"Content-Type": "text/csv"}},
              body
            }});
            const payload = await res.json();
            if (!res.ok) {{
              document.getElementById("batchSummary").textContent = payload.detail || "Batch prediction failed";
              document.getElementById("batchTable").innerHTML = "";
              return;
            }}
            renderBatch(payload.predictions || []);
          }}
          function renderBatch(rows) {{
            document.getElementById("batchSummary").textContent = `${{rows.length}} row(s) scored`;
            if (!rows.length) {{
              document.getElementById("batchTable").innerHTML = "";
              return;
            }}
            const htmlRows = rows.map(row => {{
              const risk = Number(row.predicted_churn) === 1 ? "High churn risk" : "Lower churn risk";
              return `<tr>
                <td>${{row.msno || ""}}</td>
                <td>${{(Number(row.churn_probability) * 100).toFixed(1)}}%</td>
                <td>${{risk}}</td>
                <td>${{Number(row.threshold).toFixed(3)}}</td>
              </tr>`;
            }}).join("");
            document.getElementById("batchTable").innerHTML = `
              <table>
                <thead><tr><th>User</th><th>Churn probability</th><th>Decision</th><th>Threshold</th></tr></thead>
                <tbody>${{htmlRows}}</tbody>
              </table>`;
          }}
        </script>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight health check."""

    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict[str, Any]:
    """Return scoring schema guidance and an example record."""

    try:
        artifacts = _load_artifacts()
        required_columns = _required_input_columns(artifacts.preprocessor)
    except Exception:
        required_columns = []
    return {
        "input_contract": "engineered user-level feature rows",
        "required_model_columns": required_columns,
        "example_record": EXAMPLE_RECORD,
        "notes": [
            "Ratios such as auto_renew_rate and cancel_rate should be between 0 and 1.",
            "Counts, spend, seconds, and days-since features should be non-negative.",
            "For best results, send the full engineered feature schema generated by the project pipeline.",
        ],
    }


@app.post("/predict")
def predict_json(payload: JsonScoringRequest) -> dict[str, Any]:
    """Score engineered feature rows submitted as JSON records."""

    try:
        frame = pd.DataFrame(payload.records)
        predictions = _score_frame(frame)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"rows": len(predictions), "predictions": predictions.to_dict(orient="records")}


@app.post("/predict-csv")
async def predict_csv(request: Request) -> dict[str, Any]:
    """Score engineered feature rows submitted as CSV text."""

    try:
        body = await request.body()
        frame = pd.read_csv(io.BytesIO(body))
        predictions = _score_frame(frame)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"rows": len(predictions), "predictions": predictions.to_dict(orient="records")}
