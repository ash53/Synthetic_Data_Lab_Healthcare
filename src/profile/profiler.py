import pandas as pd, numpy as np
from src.utils.io import load_tabular, save_report_html

def _html_section(title, content):
    return f"<h2>{title}</h2>\n{content}"

def profile_tabular(cfg, synth_df):
    real = load_tabular(cfg["data"]["input_csv"])
    cols = cfg["data"]["numerical"] + cfg["data"]["categorical"]
    html = "<h1>Healthcare Tabular Profile</h1>"

    def stats(df):
        return df[cols].describe(include='all').to_html()

    html += _html_section("Real Summary", stats(real))
    html += _html_section("Synthetic Summary", stats(synth_df))
    save_report_html(f"{cfg['output']['reports_dir']}/profile_tabular.html", html)

def profile_images(cfg, tensor):
    html = "<h1>Image Profile</h1><p>Sample grid saved to file.</p>"
    save_report_html(f"{cfg['output']['reports_dir']}/profile_images.html", html)
