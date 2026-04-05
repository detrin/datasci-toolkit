# Tagging

Weighted TF-IDF with per-entity Z-score normalization for entity tagging.

## WeightedTFIDF

Assigns ranked tags to entities using a weighted TF-IDF score, then normalizes per entity via Z-scores and min-max scaling.

### When to use

- **Product attribute extraction**: millions of products, each with hundreds of attribute mentions from reviews and descriptions. You want the top 5-10 tags that genuinely characterize each product. The weight column carries review helpfulness or NLP confidence, so a high-confidence mention of "waterproof" counts more than a passing reference.
- **Customer interest profiling**: a bank wants to know what each customer "is about" -- mortgage customer, travel spender, investor. Tags are merchant categories, value is transaction count, weight is transaction amount. A customer with 2 large investment transfers is more "investor" than one with 50 small coffee purchases.
- **Support ticket routing**: keywords extracted from tickets, weighted by extraction confidence. Subject-line keywords get a higher level than body text. The Z-score normalization means a 3-word ticket and a 500-word ticket both route on their most distinctive terms.
- **Ad targeting / audience segmentation**: users tagged by browsed/purchased product categories, weighted by dwell time or conversion signal. You want the top 3-5 interest tags per user, comparable across power users and casual visitors.
- **Any (entity, tag, count) problem** where you have an external quality signal and need the most *distinctive* tags per entity -- not the most frequent, not the globally rarest, but the ones that are unusually strong for that specific entity relative to its own distribution.

### Basic usage (standard TF-IDF)

```python
import polars as pl
from datasci_toolkit import WeightedTFIDF

df = pl.DataFrame({
    "doc_id": ["A", "A", "A", "B", "B", "B"],
    "term": ["python", "data", "ml", "python", "web", "api"],
    "count": [10, 8, 5, 12, 7, 3],
})

tfidf = WeightedTFIDF(score_threshold=0.1)
result = tfidf.fit_transform(df, entity_col="doc_id", tag_col="term", value_col="count")
print(result)
```

### With external weights and hierarchy

```python
df = pl.DataFrame({
    "product": ["P1", "P1", "P1", "P2", "P2"],
    "attribute": ["durable", "lightweight", "cheap", "durable", "premium"],
    "mentions": [10, 5, 20, 8, 3],
    "confidence": [0.9, 0.7, 0.3, 0.8, 0.9],
    "tier": [1.0, 1.0, 0.5, 1.0, 1.0],
})

tfidf = WeightedTFIDF(weight_col="confidence", level_col="tier")
result = tfidf.fit_transform(
    df, entity_col="product", tag_col="attribute", value_col="mentions"
)
print(result)
```

The `confidence` column weights each mention by its reliability. The `tier` column boosts primary attributes over secondary ones.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `zscore_thresh` | `2.0` | Tags with Z-score above this are dominant (score=1.0). |
| `score_threshold` | `0.1` | Minimum final score to retain a tag. |
| `weight_col` | `None` | Column with external relevance signal. None = all 1.0. |
| `level_col` | `None` | Column with hierarchy multiplier. None = all 1.0. |

### How it works

1. **Weighted TF**: `sum(weight * value) / entity_total` — normalized within each entity
2. **IDF**: `|log10(N / (1 + entity_count))|` — penalizes globally common tags
3. **Score**: `level * TF * IDF`
4. **Z-score**: per-entity normalization. Single-tag entities get Z=3.0 (dominant)
5. **Dominant tags** (Z > threshold): assigned final_score=1.0
6. **Normal tags**: min-max scaled to [0, 1] within entity, filtered by score_threshold
