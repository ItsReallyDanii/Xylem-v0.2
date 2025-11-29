import pandas as pd
import os

# --------------------------
# CONFIG
# --------------------------
BASELINE_PATH = "results/physics_validation_report.csv"  # pre-tuning
NEW_PATH = "results/physics_validation_report_post_tuned.csv"  # post-tuning

if not os.path.exists(BASELINE_PATH):
    raise FileNotFoundError("Baseline file not found. Run analyze_flow_metrics.py first.")
if not os.path.exists(NEW_PATH):
    raise FileNotFoundError("New tuned report not found. Run analyze_flow_metrics.py again after retraining.")

# --------------------------
# LOAD AND COMPARE
# --------------------------
base = pd.read_csv(BASELINE_PATH, index_col=0)
new = pd.read_csv(NEW_PATH, index_col=0)

comparison = pd.DataFrame({
    "Real Mean": base["real_mean"],
    "Synthetic (Before)": base["synthetic_mean"],
    "Synthetic (After)": new["synthetic_mean"],
    "Δ Mean Improvement": new["synthetic_mean"] - base["synthetic_mean"],
    "T-Test p Before": base["t_p_value"],
    "T-Test p After": new["t_p_value"],
    "KS-Test p Before": base["ks_p_value"],
    "KS-Test p After": new["ks_p_value"],
})

# Improvement summary
improved_t = (comparison["T-Test p After"] > comparison["T-Test p Before"]).sum()
improved_ks = (comparison["KS-Test p After"] > comparison["KS-Test p Before"]).sum()

print("\n🌿 Physics-Informed Training Effect Summary")
print("--------------------------------------------")
print(f"✅ Metrics improved (T-test): {improved_t}/{len(comparison)}")
print(f"✅ Metrics improved (KS-test): {improved_ks}/{len(comparison)}")
print("\n📊 Detailed Comparison:\n")
print(comparison.round(5))

# Save
comparison.to_csv("results/physics_comparison_report.csv")
print("\n✅ Saved detailed report → results/physics_comparison_report.csv")
