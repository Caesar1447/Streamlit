# Local API — Educational Portfolio Demonstration

This project includes a FastAPI service to demonstrate API contract design, response validation, reproducible model evaluation, and integration with a Streamlit research application.

**Not a medical API.** This is an educational engineering demonstration. It is not a medical device, a cancer screening service, or a patient risk predictor. **Do not upload or submit patient-identifying or other sensitive health information.** This service is intentionally unauthenticated and should **only run locally**. Do not expose port `8000` to the public internet.

## Install and run locally

From the repository's root directory (Python 3.11 recommended):

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Activate your virtual environment before installing packages when practical. On Windows PowerShell the activation command is `.venv\Scripts\Activate.ps1`.

Browse the automatically generated documentation at <http://127.0.0.1:8000/docs>.

## UCI Wisconsin Diagnostic benchmark (aggregate research results)

| Method | Path | Description |
|---|---|---|
| `GET` | `/benchmarks/uci/summary` | Return the fixed public dataset's source, 569 observations, 30 features, and class counts |
| `GET` | `/benchmarks/uci/evaluation` | Return deterministic train/test performance metrics, training-only 5-fold cross-validation, confusion counts, and top ten impurity-based features |

Try locally:

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/benchmarks/uci/summary').read().decode())"
```

Or navigate to <http://127.0.0.1:8000/benchmarks/uci/evaluation>. The first evaluation request can take several seconds because it trains the research model; subsequent requests are served from an in-process cache until the server restarts.

Both endpoints require **no uploaded data** and return **aggregate research metrics only**. There is no UCI individual-prediction endpoint. The positive class is *malignant*. UCI source: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>, DOI: 10.24432/C5DW2B (CC BY 4.0). The dataset is distributed with scikit-learn.

The response includes Python and scikit-learn version strings to help explain differences in metrics across deployment environments. The current dependency constraints permit more than one scikit-learn version, so scores can vary slightly across installations; record and pin dependencies before claiming exact reproducibility.

The held-out partition is an internal 80/20 stratified split. Five-fold cross-validation uses the training partition only. These results are **not external clinical validation** and cannot establish performance for real patients. The impurity-based feature ranking is not a medical explanation or evidence of causation.

## Existing synthetic-demo endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Basic service status and whether a trusted local synthetic-model artifact exists |
| `GET` | `/model/info` | Inspect the local synthetic demonstration model's features and metrics |
| `POST` | `/predict` | Predict a **synthetic label** from synthetic demonstration features only |

The synthetic `POST /predict` endpoint requires a previously generated **trusted** local model artifact. Use `python -m scripts.train_model` before testing it. Without that artifact, it intentionally returns `503` rather than trying to train itself at request time.

Do **not** use `POST /predict` on patient information. Never load third-party `joblib` artifacts that you do not trust; deserializing an untrusted model file can execute code.

## Security and deployment limitations

- No authentication, access control, audit log, or production rate limiting has been implemented.
- API responses exclude individual UCI observations, row-level probabilities, and identifying data.
- The Docker Compose example binds the API port to `127.0.0.1` on the local machine.
- API deployment to an internet-facing server requires additional security design, privacy review, and operational monitoring, which are outside this demonstration.
- The already-deployed Streamlit app is **not** expected to expose these localhost endpoints over the internet. It operates independently.

## Testing

Run from the repository root:

```bash
python -m pytest -q
```

`tests/test_api_uci.py` checks the schema and values of both UCI endpoints, verifies that no individual UCI prediction route is exposed, and checks their OpenAPI documentation.
