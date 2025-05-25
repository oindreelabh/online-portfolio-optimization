import os
from dotenv import load_dotenv
import praw
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.utils.constants import TICKERS, SUBREDDITS
from src.utils.logger import setup_logger
from src.db.write_to_neon import write_df_to_neon

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
    within the last `days` days.
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

                # Check if the submission title or selftext contains any keywords
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
            logger.error(f"Error fetching from r/{subreddit}: {e}")

    if not all_posts:
        logger.warning("No posts found matching criteria.")
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)
    return df


def main():
    subreddits = SUBREDDITS
    tickers = TICKERS

    df = fetch_recent_posts(subreddits=subreddits, keywords=tickers, limit=1000, days=1)
    if not df.empty:
        write_df_to_neon(df, "reddit_posts")
    else:
        logger.info("No new relevant Reddit posts found.")

if __name__ == "__main__":
    main()
