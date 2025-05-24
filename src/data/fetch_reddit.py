import os
from dotenv import load_dotenv
import praw
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone

from src.utils.constants import TICKERS, SUBREDDITS
from src.utils.helpers import LOG_DIR, RAW_DATA_DIR

load_dotenv()  # Load env vars from .env file

print("Logs folder is at:", LOG_DIR)
print("Raw data folder is at:", RAW_DATA_DIR)

log_file_path = os.path.join(LOG_DIR, "fetch_reddit.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)


def fetch_recent_posts(subreddits, keywords, limit=500, days=1):
    """
    Fetch recent Reddit submissions from given subreddits containing any of the keywords
    within the last `days` days.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    all_posts = []
    for subreddit in subreddits:
        logging.info(f"Fetching posts from r/{subreddit}")
        try:
            subreddit_obj = reddit.subreddit(subreddit)
            for submission in subreddit_obj.new(limit=limit):
                created = datetime.fromtimestamp(submission.created_utc, timezone.utc)
                if created < start_time:
                    continue  # Ignore older posts

                # Check if submission title or selftext contains any keywords
                text = (submission.title + " " + submission.selftext).lower()
                if any(kw.lower() in text for kw in keywords):
                    post_data = {
                        "id": submission.id,
                        "created_utc": created,
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "score": submission.score,  # upvotes minus downvotes
                        "upvote_ratio": submission.upvote_ratio,
                        "num_comments": submission.num_comments,
                        "total_awards_received": submission.total_awards_received,
                        "author": str(submission.author),
                        "url": submission.url,
                        "subreddit": subreddit,
                        "is_stickied": submission.stickied,
                        "over_18": submission.over_18,  # NSFW flag
                    }
                    all_posts.append(post_data)
        except Exception as e:
            logging.error(f"Error fetching from r/{subreddit}: {e}")

    if not all_posts:
        logging.warning("No posts found matching criteria.")
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)
    return df


def save_data(df: pd.DataFrame, output_dir=RAW_DATA_DIR):
    """
    Save Reddit data with timestamped filename.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"reddit_posts_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    logging.info(f"Saved {len(df)} Reddit posts to {file_path}")


def main():
    subreddits = SUBREDDITS
    tickers = TICKERS

    df = fetch_recent_posts(subreddits=subreddits, keywords=tickers, limit=1000, days=1)
    if not df.empty:
        save_data(df)
    else:
        logging.info("No new relevant Reddit posts found.")


if __name__ == "__main__":
    main()
