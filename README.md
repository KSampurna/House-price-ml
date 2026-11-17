# House price prediction: an end-to-end ML walkthrough

A 13-stage Jupyter notebook that builds, evaluates and packages a house-price model with scikit-learn. Each stage starts with the question it answers, explains *why* before *how*, and ends with what the real output showed.

**Business question:** predict the sale price of a house so a buyer or agent can sanity-check an asking price.

## Results (all numbers come from running the notebook)

| Model | Cross-validated MAE (training houses) | R² |
|---|---|---|
| Guess the median | $54,586 | -0.046 |
| Ridge, log price | $18,146 | 0.681 |
| Random forest, log price | $17,265 | 0.867 |
| Boosted trees, log price (default) | $16,132 | 0.879 |
| Boosted trees, log price (tuned) | $15,092 | 0.892 |

**Final test (292 houses the model never saw, scored once):** MAE $16,784, RMSE $30,796, R² 0.876. A median guess scores MAE $59,568 on the same houses.

The test score is slightly lower than the cross-validation forecast (R² 0.892). That is expected: tuning picks the best of many tries, so its own score runs a little optimistic. The test number is the honest one.

## Stages

1. Problem definition and success metric
2. Data loading
3. Exploratory data analysis
4. Data quality and cleaning
5. Feature engineering
6. Train/test split and Pipeline (no data leakage)
7. Baseline models
8. Better models with 5-fold cross-validation
9. Tuning, learning and validation curves
10. Final test evaluation and interpretation
11. Trade-off analysis (seven real experiments)
12. Deployment: saved model bundle and a `predict_price` function with a price range and warnings
13. Monitoring and drift (simulated)

## The final model

`TransformedTargetRegressor` (predicts log price, returns dollars) wrapping a `Pipeline`: a `ColumnTransformer` (median imputation with missing flags and scaling for numbers, one-hot for low-cardinality text, `TargetEncoder` for high-cardinality text) followed by `HistGradientBoostingRegressor` (266 trees, depth 2).

## How to run

```bash
pip install -U scikit-learn pandas numpy matplotlib seaborn scipy joblib jupyter
jupyter notebook house_price_ml_stages_1_to_13.ipynb
```

Run all cells from top to bottom. It takes about 2 to 3 minutes on a laptop. Tested with scikit-learn 1.9.1, pandas 3.0.5 and numpy 2.5.3. scikit-learn 1.3 or newer is required (`TargetEncoder`).

## Data

The data file is **not included**. Put your copy of `train.csv` (1,460 rows, 81 columns, target `SalePrice`) next to the notebook. It has the same shape as the Kaggle "House Prices: Advanced Regression Techniques" training file. Check the data provider's terms before redistributing it.

## Limitations

- Small dataset (1,168 training houses), so scores have real uncertainty.
- The model is least reliable above $350,000 and tends to under-price expensive houses. The `predict_price` function warns in that range.
- Stage 13 simulates drift using the held-out houses as pretend new sales. There is no live system behind it.
- The saved `.joblib` model is created when you run Stage 12 and is not committed. Joblib files can run code when loaded, so only load files you trust, and use the same scikit-learn version that saved them.
