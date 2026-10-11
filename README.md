# CreditSentinel V2

**An interactive ensemble credit-risk dashboard built with Streamlit.**

CreditSentinel V2 combines predictions from XGBoost, Random Forest and Logistic Regression in one interface. Users can select an issuer from a local cache or enter financial ratios manually, inspect individual model outputs and explore financial trends where cached history is available.

Built by [William Fernandes](https://github.com/williamf48). This is the ensemble successor to the [original CreditSentinel application](https://github.com/williamf48/Credit-Sentinel.).

## Features

- Three-model weighted ensemble with individual prediction displays.
- Manual inputs for leverage, interest coverage, promoter pledge and current ratio.
- Issuer selection from a committed local JSON cache.
- Sector-dependent leverage thresholds and transparent heuristic adjustments.
- Score gauge, summary classification and rule-based credit memo.
- Historical leverage and annual revenue charts when the selected issuer has cached records.
- Cached market capitalization and 52-week price information.

## Technology stack

| Purpose | Technologies |
| --- | --- |
| Interface | Streamlit |
| Models | XGBoost, scikit-learn |
| Data processing | Pandas, NumPy |
| Charts | Plotly |
| Model loading | XGBoost JSON, joblib |

The dependency file also includes yfinance and openpyxl. The current application reads its issuer data from `stock_cache.json`; it does not refresh that cache through a live market-data request.

## How scoring works

The application submits four features to the models:

| Input | Model feature name |
| --- | --- |
| Debt / EBITDA | `Debt_to_EBITDA` |
| Interest coverage | `Interest_Coverage` |
| Promoter pledge (%) | `Pledged_Pct` |
| Current ratio | `Current_Ratio` |

The base ensemble output is:

```text
base_probability = 0.60 × XGBoost
                 + 0.20 × Random Forest
                 + 0.20 × Logistic Regression
```

The application then adds:

- `0.20` if leverage exceeds the configured sector threshold.
- `0.10` if promoter pledge exceeds 20%.

The resulting value is clipped to `[0.01, 0.99]` and multiplied by 100 for display.

| Display score | Current UI label |
| --- | --- |
| Below 35 | INVESTMENT GRADE (IG) |
| 35 to below 70 | SPECULATIVE / HY |
| 70 and above | DISTRESSED / DEFAULT |

**Interpretation:** the final displayed value is a model-and-heuristic score. Adding penalties and clipping changes the underlying model probabilities; the adjusted value has not been demonstrated here to be a calibrated default probability. UI categories are prototype bands, not externally issued ratings or confirmed default events. Sector thresholds are application defaults, not borrower-specific contractual covenants.

## Repository contents

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and ensemble scoring |
| `creditsentinel_xgb.json` | Saved XGBoost model |
| `creditsentinel_rf.joblib` | Saved Random Forest model |
| `creditsentinel_lr.joblib` | Saved Logistic Regression pipeline |
| `stock_cache.json` | Local issuer inputs and optional financial history |
| `requirements.txt` | Python dependencies |
| `.devcontainer/devcontainer.json` | Development-container configuration |

## Run locally

### 1. Clone and create an environment

```bash
git clone https://github.com/williamf48/CreditSentinelV2.git
cd CreditSentinelV2
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies and launch

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the URL printed by Streamlit, usually http://localhost:8501.

Run from the repository root: model and cache paths are relative to the working directory. All three model artifacts must be present for startup. Load only trusted joblib artifacts.

## Using the dashboard

1. Select an issuer, or choose **Manual Entry**.
2. Review or change the four financial inputs.
3. Click **Run Ensemble Model**.
4. Inspect the aggregate score and each model's contribution.
5. Review the credit memo and available financial charts.

Issuer selection can trigger scoring without submitting the form. Missing cache data leaves manual entry available; charts depend on the optional history and revenue fields.

## Reproducibility and limitations

- The repository contains saved inference artifacts, but no training or evaluation pipeline.
- Dataset provenance, preprocessing decisions, training splits, ensemble-weight selection and probability calibration are not reproducible from the committed files.
- No evaluation metrics are claimed in this README because the repository does not include the code and results needed to verify them.
- Dependency versions are unpinned. Serialized scikit-learn models may require the package versions used during training.
- Cache freshness is not guaranteed, and no refresh script is included.
- The interface displays model contributions and rule explanations; it does not implement SHAP attribution or a persistent audit log.
- Automated tests and a documented model-validation workflow remain improvements.

For reproducible research, add dataset instructions, training code, preprocessing, package versions, held-out evaluation and validation of both the base ensemble and adjusted score.

## Related work

- [CreditSentinel original](https://github.com/williamf48/Credit-Sentinel.): single-model predecessor.
- [RiskVision](https://github.com/williamf48/RiskVision): historical deterioration monitoring, portfolio analytics and covenant analysis.

## License

No license file is currently included in this repository.

