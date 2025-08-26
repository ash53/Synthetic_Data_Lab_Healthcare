import pandas as pd

def load_tabular(path):
    return pd.read_csv(path)

def save_report_html(path, html_str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
