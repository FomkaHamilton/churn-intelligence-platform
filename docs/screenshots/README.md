# Screenshots

Screenshots of the live app running with the 2,000-customer sample dataset.

| File | Page | What it shows |
| --- | --- | --- |
| `01_upload_quality.png` | Upload | Successful ingestion + data quality report (row counts, null flags, severity badges) |
| `02_kpi_trends.png` | Analytics | MRR, active subscribers, and churn rate time-series charts |
| `03_cohort_heatmap.png` | Analytics | Cohort retention matrix — colour-coded heatmap with month-1 through month-12 retention rates |
| `04_churn_predictions.png` | Predictions | ROC curve, top-20 at-risk customer table, SHAP feature importance bar chart |
| `05_clv_segments.png` | Predictions | CLV distribution histogram + customer segment donut (New / Healthy / At-Risk / High-Value / Churned) |
| `06_forecast.png` | Forecasting | 12-month revenue + subscriber forecast with 80% confidence band |
| `07_ai_briefing.png` | Insights | AI executive briefing — five-section structured report with health colour indicator |
| `08_pdf_export.png` | Insights | PDF download confirmation + sample of the exported executive briefing |

## How to capture

With the app running locally (`streamlit run app.py`) and sample data loaded
(`Load sample dataset` button on the Upload page), take a screenshot of each
page listed above and save it here under the filename shown.

To annotate (optional):

```python
from PIL import Image, ImageDraw, ImageFont

img = Image.open("docs/screenshots/04_churn_predictions.png")
draw = ImageDraw.Draw(img)
draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
draw.text((x1, y1 - 20), "SHAP ranks recency #1 by design", fill="red")
img.save("docs/screenshots/04_churn_predictions_annotated.png")
```
