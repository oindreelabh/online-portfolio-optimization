"""
Transaction cost impact analysis.

Assumes per-date predicted returns and current allocations; computes:
- Naive target weights (rank-based, full rebalance)
- Constrained weights (using suggest_rebalance with 20% change & 25% cap)
- Turnover & cost difference

INPUT
predictions CSV: date,ticker,predicted_return
Optionally provide initial_weights CSV: ticker,weight (if omitted equal-weight first day)
Transaction cost modeled as: cost_rate * turnover_notional (turnover = 0.5 * sum|Δw| per rebalance)

OUTPUT
- tc_impact.csv
- turnover_plot.html

USAGE
python -m src.analysis.transaction_cost_impact \
  --predictions-csv evaluation_results/perf/predictions_long.csv \
  --model-name HYBRID \
  --output-dir analysis_results/transaction_cost \
  --cost-rate 0.001
"""
import os
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from src.model.lstm_ogdm_hybrid import suggest_rebalance

def rank_weights(pred_map):
    tickers = sorted(pred_map, key=pred_map.get, reverse=True)
    n = len(tickers)
    denom = n * (n + 1) / 2
    return {t: (n - i) / denom for i, t in enumerate(tickers)}

def apply_constraints(pred_map, current_allocs):
    return suggest_rebalance(pred_map, current_allocs)

def compute_turnover(prev_w, new_w):
    all_t = set(prev_w) | set(new_w)
    delta = sum(abs(new_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t)
    # 0.5 factor typical for buy+sell representation (optional)
    return 0.5 * delta

def main():
    ap = argparse.ArgumentParser(description="Analyze transaction cost impact of constraints.")
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--model-name", required=True, help="Filter predictions for this model (e.g., HYBRID)")
    ap.add_argument("--initial-weights", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cost-rate", type=float, default=0.001)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    preds = pd.read_csv(args.predictions_csv)
    preds["date"] = pd.to_datetime(preds["date"])
    if not {"date", "ticker", "model", "predicted_return"}.issubset(preds.columns):
        raise ValueError("Predictions CSV must have columns: date,ticker,model,predicted_return")

    preds = preds[preds["model"] == args.model_name].copy()
    dates = sorted(preds["date"].unique())

    if args.initial_weights and os.path.exists(args.initial_weights):
        init = pd.read_csv(args.initial_weights).set_index("ticker")["weight"].to_dict()
    else:
        first_day_tickers = preds[preds["date"] == dates[0]]["ticker"].unique()
        init = {t: 1 / len(first_day_tickers) for t in first_day_tickers}

    prev_naive = init
    prev_constrained = init
    rows = []
    for d in dates:
        day_preds = preds[preds["date"] == d].set_index("ticker")["predicted_return"].to_dict()
        naive_target = rank_weights(day_preds)
        constrained_target = apply_constraints(day_preds, prev_constrained)

        naive_turn = compute_turnover(prev_naive, naive_target)
        constr_turn = compute_turnover(prev_constrained, constrained_target)

        naive_cost = naive_turn * args.cost_rate
        constr_cost = constr_turn * args.cost_rate

        rows.append({
            "date": d,
            "naive_turnover": naive_turn,
            "constrained_turnover": constr_turn,
            "naive_cost": naive_cost,
            "constrained_cost": constr_cost,
            "turnover_reduction": naive_turn - constr_turn,
            "cost_saving": naive_cost - constr_cost
        })

        prev_naive = naive_target
        prev_constrained = constrained_target

    impact_df = pd.DataFrame(rows)
    impact_df.to_csv(os.path.join(args.output_dir, "tc_impact.csv"), index=False)

    fig = px.line(impact_df.melt(id_vars="date", value_vars=["naive_turnover", "constrained_turnover"]),
                  x="date", y="value", color="variable",
                  title="Turnover Comparison")
    fig.write_html(os.path.join(args.output_dir, "turnover_plot.html"))
    print("Saved transaction cost impact to", args.output_dir)

if __name__ == "__main__":
    main()
