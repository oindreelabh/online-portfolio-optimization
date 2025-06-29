import os
from dotenv import load_dotenv
import praw
import pandas as pd
import argparse
from datetime import datetime, timedelta, timezone

from src.data.preprocess import preprocess_reddit
from src.utils.constants import TICKERS, SUBREDDITS
from src.utils.logger import setup_logger
from src.utils.helpers import write_df_to_csv
from src.utils.ticker_dict import build_ticker_keyword_map_openfigi

load_dotenv()  # Load env vars from .env file

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def fetch_recent_posts(subreddits, ticker_mapping, limit=500, days=30):
    """
    Fetch recent Reddit submissions from given subreddits containing any of the keywords
    within the last `days` days. Concatenates title, selftext, and top 10 comments (by upvotes) into one text field.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    all_posts = []
    for subreddit in subreddits:
        logger.info(f"Fetching posts from r/{subreddit}")
        try:
            subreddit_obj = reddit.subreddit(subreddit)
            for submission in subreddit_obj.new(limit=limit):
                created = datetime.fromtimestamp(submission.created_utc, timezone.utc)
                if created < start_time:
                    continue  # Ignore older posts
                # Combine title and selftext for keyword search
                text = (submission.title + " " + submission.selftext).lower()
                keywords = [t.lower() for t in ticker_mapping.keys()] + [
                    name.lower() for names in ticker_mapping.values() for name in names]
                if any(kw.lower() in text for kw in keywords):
                    # Find which tickers are mentioned
                    mentioned_tickers = [k for k in ticker_mapping if k.lower() in text or ticker_mapping[k].lower() in text]

                    # Fetch and sort comments by upvotes (score)
                    submission.comments.replace_more(limit=0)
                    comments = submission.comments.list()
                    top_comments = sorted(
                        comments,
                        key=lambda c: getattr(c, "score", 0),
                        reverse=True
                    )[:10]
                    # Concatenate top comment bodies
                    top_comments_text = " ".join([c.body for c in top_comments])

                    # Concatenate title, selftext, and top comments
                    full_text = submission.title + "\n" + submission.selftext + "\n" + top_comments_text

                    post_data = {
                        "id": submission.id,
                        "date": created.date(),
                        "full_text": full_text,
                        "score": submission.score,
                        "upvote_ratio": submission.upvote_ratio,
                        "num_comments": submission.num_comments,
                        "total_awards_received": submission.total_awards_received,
                        "author": str(submission.author),
                        "subreddit": subreddit,
                        "tickers": mentioned_tickers,
                    }
                    all_posts.append(post_data)
        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit}: {e}")

    if not all_posts:
        logger.warning("No posts found matching criteria.")
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)
    return df

def fetch_and_store_recent_reddit_posts(
    subreddits,
    ticker_dict,
    limit,
    days,
    filename,
    raw_dir
):
    """
    Fetches recent Reddit posts and writes them to Neon DB.
    Designed for Airflow or other programmatic use.
    """
    logger.info(f"Starting fetch_and_store_recent_reddit_posts for last {days} days, limit {limit}")
    df = fetch_recent_posts(subreddits=subreddits, ticker_mapping=ticker_dict, limit=limit, days=days)
    if not df.empty:
        logger.info(f"Fetched {len(df)} Reddit posts.")
        df = preprocess_reddit(df)
        logger.info("Preprocessed Reddit posts.")
        csv_path = write_df_to_csv(df, raw_dir, filename)
        logger.info(f"Latest data written successfully to {csv_path}.")
    else:
        logger.info("No new relevant Reddit posts found.")

if __name__ == "__main__":
    logger.info("Starting Reddit data fetching and storing...")
    parser = argparse.ArgumentParser(description="Fetch and store recent Reddit posts.")
    parser.add_argument("--filename", type=str, default="reddit_posts.csv", help="Filename to save the fetched posts")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory for raw data files")
    args = parser.parse_args()

    ticker_map = build_ticker_keyword_map_openfigi(TICKERS, args.raw_dir)

    fetch_and_store_recent_reddit_posts(
        subreddits=SUBREDDITS,
        ticker_dict=ticker_map,
        limit=1000,
        days=1,
        filename=args.filename,
        raw_dir=args.raw_dir
    )
    logger.info("Reddit data fetching and storing completed.")
