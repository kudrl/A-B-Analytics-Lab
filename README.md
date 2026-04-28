# A/B Analytics Lab

Educational / portfolio project for reproducible A/B test analysis.

A Streamlit app for loading event-level experiment data, calculating core A/B metrics, running basic statistical checks, inspecting a DuckDB SQL trace, and exporting reproducible reports.

## Features

- CSV upload / synthetic data
- SRM check
- CR / ARPU / retention / funnel
- z-test for conversion
- bootstrap CI for ARPU
- DuckDB SQL trace
- export report

## Input schema

Required columns:

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | string / integer | User identifier |
| `variant` | string | Experiment group: `A` or `B` |
| `ts` | datetime | Event timestamp |
| `event` | string | Event name, for example `signup`, `open_app`, `pay` |

Optional columns:

| Column | Type | Description |
| --- | --- | --- |
| `amount` | number | Revenue amount for payment events. Missing values are treated as `0`. |

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Example

1. Start the app.
2. Use synthetic data or upload a CSV with the schema above.
3. Choose the payment event name.
4. Review KPIs, SRM, conversion test, ARPU bootstrap, retention, and funnel.
5. Export individual files or `ab_results.zip`.

## Limitations

- only two variants A/B
- bootstrap p-value is approximate
- funnel is event-order based
- no sequential testing correction
- no CUPED / stratification yet
