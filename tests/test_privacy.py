from src.privacy.membership_inference import run_membership_inference
import pandas as pd, os, json

def test_membership(tmp_path):
    df = pd.DataFrame({
        "age":[30,40,50,60], "bmi":[20,22,28,31],
        "systolic_bp":[120,130,140,145], "diastolic_bp":[80,82,90,92],
        "cholesterol":[180,200,220,240],
        "sex":["F","M","F","M"], "smoker":[0,1,0,1],
        "diagnosis_diabetes":[0,0,1,1]
    })
    out_dir = tmp_path / "rep"
    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        "data":{"input_csv": str(tmp_path/'in.csv'), "target":"diagnosis_diabetes",
                "numerical":["age","bmi","systolic_bp","diastolic_bp","cholesterol"],
                "categorical":["sex","smoker"]},
        "output":{"reports_dir": str(out_dir)},
    }
    df.to_csv(cfg["data"]["input_csv"], index=False)
    run_membership_inference(cfg, df.copy())
    assert (out_dir / "membership.json").exists()
