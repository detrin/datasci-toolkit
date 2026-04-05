# Smoothing

Adaptive temporal smoothing for count time series and classification probabilities.

## PoissonSmoother

Smooths count time series using Poisson CDF p-values as adaptive blend weights. When today's count is consistent with history, the output is smoothed toward the historical mean. When it's anomalous, the raw observation is trusted.

### When to use

- **Credit scoring**: transaction counts per month fluctuate randomly. Smooth them before feature engineering so "num_transactions_last_6mo" reflects the customer's true activity level, not a lucky/unlucky month.
- **Fraud detection**: login attempt counts spike during attacks but also vary day-to-day. Poisson smoothing preserves genuine attack spikes (low p-value → trust today) while dampening normal variance.
- **E-commerce**: daily order counts per product are noisy. Smoothing gives demand planners a stable baseline while letting flash-sale volumes pass through untouched.
- **IoT/sensor data**: event trigger counts per device per hour. Normal variance gets smoothed, genuine anomalies (device failure, environmental event) are preserved for alerting.

### Basic usage

```python
import polars as pl
from datasci_toolkit import PoissonSmoother

df = pl.DataFrame({
    "product_id": ["A"] * 7,
    "date": [f"2024-01-{d:02d}" for d in range(1, 8)],
    "orders": [50, 48, 52, 49, 51, 50, 320],
})

ps = PoissonSmoother(window_size=7, alpha=0.1).fit()
result = ps.transform(
    df,
    entity_cols=["product_id"],
    date_col="date",
    value_col="orders",
    target_date="2024-01-07",
)
print(result)
```

The flash-sale count of 320 will be mostly trusted (low p-value), while a normal count like 48 would be smoothed toward ~50.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `window_size` | `7` | Number of time periods including target. Min 2. |
| `alpha` | `0.1` | Weight compression exponent. Lower = more smoothing. |
| `eps_left` | `20` | CDF stabilization for lower tail. |
| `eps_right` | `5` | CDF stabilization for upper tail. |

### How it works

1. Splits data into today (target_date) and history
2. Computes historical mean per entity
3. Calculates Poisson p-value: how likely is today's count given the historical mean?
4. Blends: `smoothed = w_today * today + w_hist * history_sum`
5. Filters out rows where smoothed_count <= 0

---

## PredictionSmoother

Averages classification probabilities across time periods to stabilize label assignments. Reduces label churn caused by small feature fluctuations near decision boundaries.

### When to use

- **Credit risk monitoring**: a model re-scores customers monthly. Small feature movements cause predictions to oscillate around the 0.5 threshold -- a customer flips from "low risk" to "medium risk" and back, triggering unnecessary re-pricing or agent outreach. Smoothing over 3-6 months eliminates the churn.
- **Customer segmentation**: a classification model assigns customers to segments (high-value, growth, at-risk). Month-to-month jitter in segment labels breaks downstream reporting and campaign targeting. Smoothing stabilizes the labels.
- **Churn prediction**: a model scores users monthly. Smoothing prevents false churn alerts from one-off dips in engagement probability.
- **Any production model scored periodically** where label stability matters more than instant reactivity.

### Binary mode

```python
import polars as pl
from datasci_toolkit import PredictionSmoother

df = pl.DataFrame({
    "customer_id": ["C1", "C1", "C1"],
    "month": [1, 2, 3],
    "prob_default": [0.52, 0.48, 0.51],
})

ps = PredictionSmoother(min_observations=2).fit()
result = ps.transform(
    df,
    entity_cols=["customer_id"],
    period_col="month",
    prob_cols="prob_default",
)
print(result)
```

### Multi-class mode

```python
df = pl.DataFrame({
    "product_id": ["P1", "P1", "P1"],
    "month": [1, 2, 3],
    "prob_electronics": [0.51, 0.48, 0.53],
    "prob_clothing": [0.39, 0.42, 0.37],
    "prob_food": [0.10, 0.10, 0.10],
})

ps = PredictionSmoother().fit()
result = ps.transform(
    df,
    entity_cols=["product_id"],
    period_col="month",
    prob_cols=["prob_electronics", "prob_clothing", "prob_food"],
)
print(result)
```

Multi-class mode adds a `predicted_label` column with the column name having the highest averaged probability.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_observations` | `1` | Minimum time periods required to produce output. |
