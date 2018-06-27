"""
Scrape comments from reddit using PRAW
"""

import praw
import pickle

# Choose subreddit
subR = 'Fitness'

reddit = praw.Reddit(client_id='R9SP3Yf3rXw0zQ',
                     client_secret='ZHCRBo6-JACS2mf87goAtqZxSP8',
                     username='FlyKillaDataGrl',
                     password='5ASROCKs',
                     user_agent='praw_comment_scraper')

# Choose subreddit
subreddit = reddit.subreddit(subR)

# Choose tab in subreddit (hot, rising, controversial, top, etc.)
controversial_mensrights = subreddit.controversial()

all_comments = []

# Go through data and get comments
for submission in controversial_mensrights:
    if not submission.stickied:
        all_comments.append(submission.title)

        comments = submission.comments.list()
        for comment in comments:

            try:
                all_comments.append(comment.body)
            except AttributeError:
                pass

# View number of comments scraped
print(len(all_comments))

# Pickle data, filename = subreddit + # of comments scraped
with open(f'{subR}_{len(all_comments)}.pkl', 'wb') as f:
    pickle.dump(all_comments, f)