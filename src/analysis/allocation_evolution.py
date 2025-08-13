"""
Allocation evolution analysis.

INPUT
weights CSV: date,ticker,weight

OUTPUT
- allocation_summary.csv (date, herfindahl, top_weight, num_assets)
- allocation_evolution.html (stacked area)
- concentration.html (herfindahl & top weight time series)

USAGE
python -m src.analysis.allocation_evolution \
  --weights-csv evaluation_results/perf/hybrid_weights.csv \
  --output-dir analysis_results/alloc_evolution \
  --min-weight 0.001
"""
import os
import argparse
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def load_weights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def compute_concentration(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for d, grp in df.groupby("date"):
        w = grp["weight"].values
        herf = np.sum(w ** 2)
        recs.append({
            "date": d,
            "herfindahl": herf,
            "top_weight": w.max(),
            "num_assets": (w > 0).sum()
        })
    return pd.DataFrame(recs)

def make_area_chart(df: pd.DataFrame, out_path: str) -> None:
    pivot = df.pivot(index="date", columns="ticker", values="weight").fillna(0)
    pivot = pivot.sort_index()
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col],
            stackgroup="one", name=col,
            mode="lines", line=dict(width=1)
        ))
    fig.update_layout(title="Allocation Evolution (Stacked Area)",
                      yaxis_title="Weight",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.write_html(out_path)

def make_concentration_plot(conc: pd.DataFrame, out_path: str) -> None:
    fig = px.line(conc.melt(id_vars="date"), x="date", y="value", color="variable",
                  title="Concentration Metrics (Herfindahl & Top Weight)")
    fig.write_html(out_path)

def main():
    ap = argparse.ArgumentParser(description="Analyze allocation evolution.")
    ap.add_argument("--weights-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-weight", type=float, default=0.0, help="Filter out tiny weights for clarity in plot.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    w = load_weights(args.weights_csv)
    if args.min_weight > 0:
        w = w[w["weight"] >= args.min_weight]

    conc = compute_concentration(w)
    conc.to_csv(os.path.join(args.output_dir, "allocation_summary.csv"), index=False)
    make_area_chart(w, os.path.join(args.output_dir, "allocation_evolution.html"))
    make_concentration_plot(conc[["date", "herfindahl", "top_weight"]], os.path.join(args.output_dir, "concentration.html"))
    print("Saved allocation evolution analysis to", args.output_dir)

if __name__ == "__main__":
    main()
