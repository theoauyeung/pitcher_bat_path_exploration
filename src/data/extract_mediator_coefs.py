"""Extract mediator model coefficients + 95% CI to CSV for R gt table."""
import sys
import types
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec

# Intercept xgboost/sklearn/rich imports before they fire so joblib can unpickle
# the statsmodels mediator objects without hitting the rich -> network-timeout chain.
class _StubModule(types.ModuleType):
    """Module stub that auto-creates class placeholders for any attribute."""
    __path__ = []
    def __getattr__(self, name):
        cls = type(name, (), {"__init__": lambda s, *a, **kw: None,
                              "__reduce__": lambda s: (type(s), ())})
        setattr(self, name, cls)
        return cls

class _StubLoader(Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)
    def exec_module(self, module):
        pass

class _StubFinder(MetaPathFinder):
    _STUB = ("xgboost", "sklearn", "rich")
    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self._STUB):
            return ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None

sys.meta_path.insert(0, _StubFinder())

import joblib
import numpy as np
import pandas as pd

print("Loading causal_models.joblib ...")
models = joblib.load("models/causal_models.joblib")["mediator_models"]

COMMIT_MS = 150
DEV_X = f"pc{COMMIT_MS}_emp_dev_x"
DEV_Z = f"pc{COMMIT_MS}_emp_dev_z"

PARAM_LABELS = {
    "Intercept":                       "Intercept",
    DEV_X:                             "H. Displacement",
    DEV_Z:                             "V. Displacement",
    f"pc{COMMIT_MS}_emp_x_proj":       "Proj. plate_x",
    f"pc{COMMIT_MS}_emp_z_proj":       "Proj. plate_z",
    "release_speed":                   "Release Speed",
    "balls":                           "Balls",
    "strikes":                         "Strikes",
    "offset_y_ms":                     "Timing Offset",
}

RESP_LABELS = {
    "vert_attack_angle_dev": "VAA",
    "horz_attack_angle_dev": "HAA",
    "swing_path_tilt_dev":   "Tilt",
}

rows = []
for resp, model in models.items():
    ci = model.conf_int()
    for param, label in PARAM_LABELS.items():
        if param not in model.params:
            continue
        rows.append({
            "param_label": label,
            "resp_label":  RESP_LABELS[resp],
            "coef":        float(model.params[param]),
            "lower":       float(ci.loc[param, 0]),
            "upper":       float(ci.loc[param, 1]),
        })

df = pd.DataFrame(rows)
df.to_csv("results/tables/mediator_coefficients.csv", index=False)
print("Saved results/tables/mediator_coefficients.csv")
print(df.to_string(index=False))
