"""
Automated alert system for sending weekly or monthly market and sentiment summaries via email.

- Reads a single processed CSV (expects at least: date, ticker, close, and a sentiment column provided via --sentiment_col).
- Aggregates metrics over the selected period and flags significant events.
- Sends an HTML/text email via SMTP with TLS.
"""

import os
import argparse
import smtplib
from typing import List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils.logger import setup_logger
from src.utils.constants import TICKERS

# Initializing logger and environment
load_dotenv()
logger = setup_logger(os.path.basename(__file__).replace(".py", ""))


def parse_recipients(email_to_arg: Optional[str]) -> List[str]:
    """Parse comma-separated recipients from CLI or EMAIL_TO env."""
    email_to_raw = email_to_arg or os.getenv("EMAIL_TO", "")
    return [x.strip() for x in email_to_raw.split(",") if x.strip()]


def read_market_data(yf_path: str) -> pd.DataFrame:
    """Read market CSV and ensure columns: date, ticker, close."""
    df = pd.read_csv(yf_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Market CSV must contain 'date'.")
    if "ticker" not in df.columns:
        symbol_col = next((c for c in df.columns if c in {"symbol", "security", "asset"}), None)
        if symbol_col:
            df = df.rename(columns={symbol_col: "ticker"})
        else:
            raise ValueError("Market CSV must contain 'ticker'.")
    if "close" not in df.columns:
        alt = next((c for c in df.columns if c in {"adj_close", "adjusted_close", "close_price"}), None)
        if not alt:
            raise ValueError("Market CSV must contain 'close' (or adj_close).")
        df = df.rename(columns={alt: "close"})
    # Ensure timezone-aware (UTC) to avoid tz-naive/aware comparison issues
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["ticker"].isin(TICKERS)].copy()
    return df


def aggregate_sentiment(df: pd.DataFrame, period: str, sentiment_col: str) -> pd.DataFrame:
    """Aggregate mean sentiment and sample count per ticker in the window using the specified sentiment column."""
    s_col = sentiment_col.strip().lower()
    if s_col not in df.columns:
        raise ValueError(f"Sentiment column '{sentiment_col}' not found in data.")
    start_time, end_time = compute_period_window(period)
    dfp = df.copy()
    # Ensure date is datetime and filtered to window
    dfp["date"] = pd.to_datetime(dfp["date"], utc=True)
    dfp = dfp[(dfp["date"] >= start_time) & (dfp["date"] <= end_time)]
    if dfp.empty:
        return pd.DataFrame(columns=["ticker", "mean_sentiment", "samples"])
    # Drop NaNs in sentiment column before aggregation
    dfp = dfp.dropna(subset=[s_col])
    if dfp.empty:
        return pd.DataFrame(columns=["ticker", "mean_sentiment", "samples"])
    return (
        dfp.groupby("ticker")
        .agg(mean_sentiment=(s_col, "mean"), samples=(s_col, "count"))
        .reset_index()
    )


def compute_period_window(period: str) -> Tuple[datetime, datetime]:
    """Compute [start, end] window for 'weekly' or 'monthly'."""
    end_time = datetime.now(timezone.utc)
    if period.lower() == "weekly":
        start_time = end_time - timedelta(days=7)
    elif period.lower() == "monthly":
        start_time = end_time - timedelta(days=30)
    else:
        raise ValueError("period must be 'weekly' or 'monthly'")
    return start_time, end_time


def compute_period_returns(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Compute percentage return per ticker over the window."""
    start_time, _ = compute_period_window(period)
    rows = []
    for ticker, g in df.groupby("ticker"):
        g_sorted = g.sort_values("date")
        latest_row = g_sorted.iloc[-1]
        idx = g_sorted["date"].searchsorted(start_time, side="left")
        baseline_row = g_sorted.iloc[idx] if idx < len(g_sorted) else g_sorted.iloc[0]
        latest_close = float(latest_row["close"])
        baseline_close = float(baseline_row["close"])
        if baseline_close <= 0 or np.isnan(baseline_close) or np.isnan(latest_close):
            continue
        period_return = (latest_close - baseline_close) / baseline_close
        rows.append({
            "ticker": ticker,
            "period_return": period_return,
            "latest_close": latest_close,
            "baseline_close": baseline_close,
            "latest_date": latest_row["date"]
        })
    out = pd.DataFrame(rows)
    return out.sort_values("period_return", ascending=False)


def join_metrics(returns_df: pd.DataFrame, sentiment_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Left-join returns with sentiment metrics."""
    if sentiment_df is None or sentiment_df.empty:
        df = returns_df.copy()
        df["mean_sentiment"] = np.nan
        df["samples"] = 0
        return df
    df = returns_df.merge(sentiment_df, on="ticker", how="left")
    df["samples"] = df["samples"].fillna(0).astype(int)
    return df


def extract_significant_events(
    metrics_df: pd.DataFrame,
    period: str,
    threshold_return: Optional[float],
    threshold_sentiment: float
) -> dict:
    """Identify significant gainers/losers and sentiment extremes."""
    thr_ret = threshold_return if threshold_return is not None else (0.05 if period == "weekly" else 0.10)
    df = metrics_df.copy()
    top_gainers = df[df["period_return"] >= thr_ret].sort_values("period_return", ascending=False)
    top_losers = df[df["period_return"] <= -thr_ret].sort_values("period_return", ascending=True)
    pos_sentiment = df[df["mean_sentiment"] >= threshold_sentiment].sort_values("mean_sentiment", ascending=False)
    neg_sentiment = df[df["mean_sentiment"] <= -threshold_sentiment].sort_values("mean_sentiment", ascending=True)
    return {
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "pos_sentiment": pos_sentiment,
        "neg_sentiment": neg_sentiment
    }


def format_percentage(x: float) -> str:
    """Format number as signed percentage or N/A."""
    return "N/A" if pd.isna(x) else f"{x:+.2%}"


def build_html_table(df: pd.DataFrame, cols: List[Tuple[str, str]]) -> str:
    """Build a simple HTML table using selected columns."""
    if df.empty:
        return "<p>None</p>"
    header = "<tr>" + "".join([f"<th style='text-align:left;padding:4px'>{h}</th>" for _, h in cols]) + "</tr>"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c, _ in cols:
            val = r[c]
            if "return" in c:
                val = format_percentage(val)
            elif "sentiment" in c and not pd.isna(val):
                val = f"{val:+.3f}"
            cells.append(f"<td style='padding:4px'>{val}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table border='1' cellspacing='0' cellpadding='0' style='border-collapse:collapse'>{header}{''.join(rows)}</table>"


def compose_email_content(
    period: str,
    metrics_df: pd.DataFrame,
    events: dict,
    subject_prefix: str
) -> Tuple[str, str, str]:
    """Compose subject, plain-text, and HTML email bodies."""
    start_time, end_time = compute_period_window(period)
    subject = f"{subject_prefix} {period.capitalize()} Market Alert ({start_time.date()} to {end_time.date()})"

    text_lines = [
        f"{period.capitalize()} Market & Sentiment Alert",
        f"Window: {start_time.date()} to {end_time.date()}",
        "",
        f"Significant gainers: {len(events['top_gainers'])}",
        f"Significant losers: {len(events['top_losers'])}",
        f"Positive sentiment: {len(events['pos_sentiment'].dropna(subset=['mean_sentiment']))}",
        f"Negative sentiment: {len(events['neg_sentiment'].dropna(subset=['mean_sentiment']))}",
        "",
        "Top movers (by return):"
    ]
    top5 = metrics_df.sort_values("period_return", ascending=False).head(5)
    for _, r in top5.iterrows():
        text_lines.append(f"- {r['ticker']}: {format_percentage(r['period_return'])} | sentiment={r.get('mean_sentiment', np.nan):+.3f}")
    text_body = "\n".join(text_lines)

    html_body = "\n".join([
        f"<h3>{period.capitalize()} Market & Sentiment Alert</h3>",
        f"<p>Window: <b>{start_time.date()}</b> to <b>{end_time.date()}</b></p>",
        "<h4>Top Gainers</h4>",
        build_html_table(events["top_gainers"][["ticker", "period_return", "mean_sentiment"]], [
            ("ticker", "Ticker"), ("period_return", "Return"), ("mean_sentiment", "Sentiment")
        ]),
        "<h4>Top Losers</h4>",
        build_html_table(events["top_losers"][["ticker", "period_return", "mean_sentiment"]], [
            ("ticker", "Ticker"), ("period_return", "Return"), ("mean_sentiment", "Sentiment")
        ]),
        "<h4>Positive Sentiment</h4>",
        build_html_table(events["pos_sentiment"][["ticker", "mean_sentiment", "period_return"]], [
            ("ticker", "Ticker"), ("mean_sentiment", "Sentiment"), ("period_return", "Return")
        ]),
        "<h4>Negative Sentiment</h4>",
        build_html_table(events["neg_sentiment"][["ticker", "mean_sentiment", "period_return"]], [
            ("ticker", "Ticker"), ("mean_sentiment", "Sentiment"), ("period_return", "Return")
        ])
    ])
    return subject, text_body, html_body


def send_email(
    subject: str,
    text_body: str,
    html_body: str,
    recipients: List[str],
    email_from: Optional[str] = None,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None
) -> None:
    """Send email via SMTP with TLS using provided or env credentials."""
    email_from = email_from or os.getenv("EMAIL_FROM")
    smtp_host = smtp_host or os.getenv("SMTP_HOST")
    smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
    smtp_user = smtp_user or os.getenv("SMTP_USER")
    smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")

    if not email_from:
        raise ValueError("EMAIL_FROM is required.")
    if not recipients:
        raise ValueError("At least one recipient is required (EMAIL_TO or --email_to).")
    if not (smtp_host and smtp_port and smtp_user and smtp_password):
        raise ValueError("SMTP credentials are required (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD).")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(email_from, recipients, msg.as_string())
        logger.info(f"Email sent to {len(recipients)} recipient(s).")
    except Exception as exc:
        logger.error(f"Failed to send email: {exc}")
        raise


def write_report_csv(metrics_df: pd.DataFrame, output_dir: Optional[str], period: str) -> Optional[str]:
    """Optionally write the metrics report CSV to output_dir; return path if written."""
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"alerts_{period}_{ts}.csv")
    try:
        metrics_df.to_csv(out_path, index=False)
        logger.info(f"Alert report written to {out_path}")
        return out_path
    except Exception as exc:
        logger.warning(f"Failed to write alert report: {exc}")
        return None


def main():
    """CLI entrypoint for generating and sending alerts."""
    parser = argparse.ArgumentParser(description="Send weekly/monthly market & sentiment alerts via email.")
    parser.add_argument("--data_path", required=True, help="Path to processed CSV (date, ticker, close, and sentiment column).")
    parser.add_argument("--sentiment_col", required=True, help="Name of the sentiment column to aggregate.")
    parser.add_argument("--period", choices=["weekly", "monthly"], default="weekly", help="Alert period.")
    parser.add_argument("--threshold_return", type=float, help="Return threshold (5% weekly, 10% monthly by default).")
    parser.add_argument("--threshold_sentiment", type=float, default=0.3, help="Abs sentiment threshold.")
    parser.add_argument("--subject_prefix", default="Portfolio", help="Subject prefix.")
    parser.add_argument("--email_to", help="Recipients, comma-separated. Overrides EMAIL_TO.")
    parser.add_argument("--email_from", help="From address. Overrides EMAIL_FROM.")
    parser.add_argument("--smtp_host", help="SMTP host. Overrides SMTP_HOST.")
    parser.add_argument("--smtp_port", type=int, help="SMTP port. Overrides SMTP_PORT.")
    parser.add_argument("--smtp_user", help="SMTP user. Overrides SMTP_USER.")
    parser.add_argument("--smtp_password", help="SMTP password. Overrides SMTP_PASSWORD.")
    parser.add_argument("--output_dir", help="Optional directory to save CSV alert report.")
    args = parser.parse_args()

    logger.info(f"Loading processed data from {args.data_path}")
    market_df = read_market_data(args.data_path)  # ensures date/ticker/close, lowercases columns, filters TICKERS

    returns_df = compute_period_returns(market_df, args.period)
    sentiment_agg = aggregate_sentiment(market_df, args.period, args.sentiment_col)

    metrics_df = join_metrics(returns_df, sentiment_agg)

    events = extract_significant_events(
        metrics_df,
        period=args.period,
        threshold_return=args.threshold_return,
        threshold_sentiment=args.threshold_sentiment
    )

    subject, text_body, html_body = compose_email_content(
        period=args.period,
        metrics_df=metrics_df,
        events=events,
        subject_prefix=args.subject_prefix
    )

    # Always write CSV report if output_dir provided (useful for audit/logging)
    write_report_csv(metrics_df, args.output_dir, args.period)

    # Resolve recipients and send the email (must have SMTP/env properly set)
    recipients = parse_recipients(args.email_to)
    send_email(
        subject=subject,
        text_body=text_body,
        html_body=html_body,
        recipients=recipients,
        email_from=args.email_from,
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_user=args.smtp_user,
        smtp_password=args.smtp_password
    )


if __name__ == "__main__":
    # Single entrypoint. Removed duplicate CLI block to avoid double execution.
    main()
