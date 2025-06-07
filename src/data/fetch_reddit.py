import os
from dotenv import load_dotenv
import praw
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.data.preprocess import preprocess_reddit
from src.utils.constants import TICKERS, SUBREDDITS
from src.utils.logger import setup_logger
from src.utils.helpers import write_df_to_csv

load_dotenv()  # Load env vars from .env file

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def fetch_recent_posts(subreddits, keywords, limit=500, days=1):
    """
    Fetch recent Reddit submissions from given subreddits containing any of the keywords
    within the last `days` days. Adds a column for mentioned tickers.
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
                if any(kw.lower() in text for kw in keywords):
                    # Find which tickers are mentioned
                    mentioned_tickers = [kw for kw in keywords if kw.lower() in text]
                    post_data = {
                        "id": submission.id,
                        "created_utc": created,
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "score": submission.score,
                        "upvote_ratio": submission.upvote_ratio,
                        "num_comments": submission.num_comments,
                        "total_awards_received": submission.total_awards_received,
                        "author": str(submission.author),
                        "subreddit": subreddit,
                        "tickers": mentioned_tickers
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
    subreddits=SUBREDDITS,
    keywords=TICKERS,
    limit=1000,
    days=1
):
    """
    Fetches recent Reddit posts and writes them to Neon DB.
    Designed for Airflow or other programmatic use.
    """
    logger.info(f"Starting fetch_and_store_recent_reddit_posts for last {days} days, limit {limit}")
    df = fetch_recent_posts(subreddits=subreddits, keywords=keywords, limit=limit, days=days)
    if not df.empty:
        logger.info(f"Fetched {len(df)} Reddit posts.")
        df = preprocess_reddit(df)
        logger.info("Preprocessed Reddit posts.")
        csv_path = write_df_to_csv(df, "../../data/raw", "reddit_posts.csv")
        logger.info(f"Latest data written successfully to {csv_path}.")
    else:
        logger.info("No new relevant Reddit posts found.")

if __name__ == "__main__":
    # Example usage
    fetch_and_store_recent_reddit_posts(
        subreddits=SUBREDDITS,
        keywords=TICKERS,
        limit=1000,
        days=1
    )
    logger.info("Reddit data fetching and storing completed.")
