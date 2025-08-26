import argparse, yaml, os, json
from src.utils.seed import set_seed
from src.profile.profiler import profile_tabular, profile_images
from src.eval.metrics import evaluate_tabular, evaluate_images
from src.synth.tabular_gan import train_tabular_gan, generate_tabular
from src.synth.image_vae import train_image_vae, generate_images
from src.synth.ctgan import train_ctgan, generate_ctgan
from src.privacy.membership_inference import run_membership_inference

REQUIRED_KEYS = [
    "task",
    "data.input_csv",
    "data.target",
    "data.categorical",
    "data.numerical",
    "model.type",
    "train.epochs",
    "train.batch_size",
]

def _require(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        if part not in cur:
            raise KeyError(f"Missing '{dotted}' in config. Check indentation and spelling in your YAML.")
        cur = cur[part]
    return cur

def validate_config(cfg):
    for k in REQUIRED_KEYS:
        _require(cfg, k)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    validate_config(cfg)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)

    set_seed(cfg.get("seed", 0))

    task = cfg["task"]
    if task == "tabular":
        model_type = cfg["model"]["type"]
        if model_type == "gan":
            g_state, meta = train_tabular_gan(cfg)
            synth_df = generate_tabular(cfg, g_state, meta)
        elif model_type == "ctgan":
            g_state, meta = train_ctgan(cfg)
            overrides = cfg.get("sampling", {}).get("overrides", None)
            synth_df = generate_ctgan(cfg, g_state, meta, overrides=overrides)
            
            # Fail-safe: if the synthetic label is single-class, try again with 50/50
            y_col = cfg["data"]["target"]
            if synth_df[y_col].nunique() < 2:
                print("[CLI] Detected single-class synthetic labels; regenerating with 50/50.")
                balanced = {"0": 0.5, "1": 0.5}
                synth_df = generate_ctgan(cfg, g_state, meta, overrides={y_col: balanced})

        else:
            raise ValueError(f"Unknown tabular model type: {model_type}")

        synth_df.to_csv(cfg["output"]["synthetic_csv"], index=False)

        profile_tabular(cfg, synth_df)
        evaluate_tabular(cfg, synth_df)

        pa = cfg.get("privacy_attack", {})
        if pa.get("enabled", False):
            run_membership_inference(cfg, synth_df)

    elif task == "image":
        vae = train_image_vae(cfg)
        synth = generate_images(cfg, vae)
        profile_images(cfg, synth)
        evaluate_images(cfg, synth)

    else:
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    main()
