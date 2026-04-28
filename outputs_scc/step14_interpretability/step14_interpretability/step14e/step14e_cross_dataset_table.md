| Dataset | Scope | AUC | AP | Macro-F1 | Weighted-F1 | Loc mAP@0.5 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| UCF-Crime (primary) | train/test in-domain | 0.927381 | 0.867482 | 0.302860 | 0.737390 | 0.012412 | mAP@0.3=0.033624, mAP@0.7=0.011200 |
| XD-Violence (Step11) | zero-shot transfer | 0.835726 | 0.884683 | 0.360842 | 0.494240 | NA | 4-class overlap F1 (fighting/shooting/explosion/abuse) |
| RWF-2000 (Step12) | fight validation (zero-shot) | 0.862250 | 0.818835 | 0.245614 | NA | NA | fight precision=1.000000, recall=0.140000, F1=0.245614 |
| ShanghaiTech (Step13) | binary robustness (zero-shot) | 0.370528 | 0.216516 | NA | NA | NA | binary anomaly only |
