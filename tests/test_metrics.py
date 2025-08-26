from src.eval.metrics import evaluate_tabular
import pandas as pd, os, json

def test_evaluate_tabular(tmp_path):
    df = pd.DataFrame({
        "age":[30,40,50,60], "bmi":[20,22,28,31],
        "systolic_bp":[120,130,140,145], "diastolic_bp":[80,82,90,92],
        "cholesterol":[180,200,220,240],
        "sex":["F","M","F","M"], "smoker":[0,1,0,1],
        "diagnosis_diabetes":[0,0,1,1]
    })
    in_csv = tmp_path / "in.csv"
    out_dir = tmp_path / "rep"
    df.to_csv(in_csv, index=False)
    cfg = {
        "data":{"input_csv": str(in_csv), "target":"diagnosis_diabetes",
                "numerical":["age","bmi","systolic_bp","diastolic_bp","cholesterol"],
                "categorical":["sex","smoker"]},
        "output":{"reports_dir": str(out_dir)},
    }
    os.makedirs(out_dir, exist_ok=True)
    evaluate_tabular(cfg, df.copy())
    assert (out_dir / "eval.json").exists()
