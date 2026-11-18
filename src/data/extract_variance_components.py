"""Extract Bayesian variance component posteriors from intention_result.joblib.
Writes results/tables/variance_components.csv for the R gt table in intention_model.ipynb.
"""
import joblib
import numpy as np
import pandas as pd

print("Loading intention_result.joblib ...")
result = joblib.load("models/intention_result.joblib")
idata_dict = result["idata"]

SIGMA_VARS = {
    "sigma":                        "Residual",
    "1|batter_id_sigma":            "Batter intercept",
    "scale(strikes)|batter_id_sigma": "Batter x Strikes slope",
    "1|pitcher_id_sigma":           "Pitcher intercept",
}

RESP_LABELS = {
    "vert_attack_angle": "VAA",
    "horz_attack_angle": "HAA",
    "swing_path_tilt":   "Tilt",
    "bat_speed":         "Bat Speed",
    "swing_length":      "Swing Length",
}

rows = []
for resp, idata in idata_dict.items():
    post = idata.posterior
    for var, label in SIGMA_VARS.items():
        if var not in post:
            print(f"  WARNING: '{var}' not found for {resp}")
            continue
        samples = post[var].values.flatten()
        rows.append({
            "param_label": label,
            "response":    resp,
            "resp_label":  RESP_LABELS.get(resp, resp),
            "mean":        float(np.mean(samples)),
            "lower":       float(np.percentile(samples, 2.5)),
            "upper":       float(np.percentile(samples, 97.5)),
        })

df = pd.DataFrame(rows)

# pivot to wide: one row per param, columns per response
df_wide = df.pivot_table(
    index="param_label", columns="resp_label",
    values=["mean", "lower", "upper"],
    aggfunc="first"
).round(3)

# flatten and reorder
df_wide.columns = [f"{resp}_{stat}" for stat, resp in df_wide.columns]
df_wide = df_wide.reset_index()

# also save long format for easier R reading
df_long = df[["param_label", "resp_label", "mean", "lower", "upper"]]
df_long.to_csv("results/tables/variance_components.csv", index=False)
print("Saved results/tables/variance_components.csv")
print(df_long.to_string(index=False))
