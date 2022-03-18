"""Get news from different sources"""
from datetime import datetime as dt

from psaw.PushshiftAPI import PushshiftAPI

from core.database import Database
import core.scanner as sn


"""Get news from different sources"""
# make sure to have the latest symbols in the database
sn.GetAssets()


def capture_reddit(subsources, start_date):
    """Get news from Reddit

    # API: reddit.com/dev/api
    # Other API: pushshift.io
    # Pushshift wrapper: psaw.readthedocs.io

    :param str subsources: i.e. wallstreetbets
    :param str start_date: start date i.e. '2/19/2021'

    """

    filters = ['url', 'title', 'subreddit']
    limit = 1000
    date_dt = dt.strptime(start_date, "%m/%d/%Y")
    date_int = int(date_dt.timestamp())

    # pull submissions from reddit
    api = PushshiftAPI()
    submitted = list(api.search_submissions(after=date_int,
                                            subreddit=subsources,
                                            filter=filters,
                                            limit=limit))

    # pull list of legit symbols from database to compare with submissions
    with Database() as db:
        query = "SELECT id, symbol FROM symbols"
        rows = db.query(query)
    symbols = {f"${row['symbol']}": row['id'] for row in rows}

    args = []
    for submit in submitted:
        words = submit.title.split()

        def extract(word):
            return word.lower().startswith('$')

        cashtags = list(set(filter(extract, words)))
        if len(cashtags) == 0:
            continue
        for cashtag in cashtags:
            if cashtag not in symbols:
                continue
            submit_time = dt.fromtimestamp(submit.created_utc).isoformat()
            arg = (submit_time, symbols[cashtag], submit.title,
                   subsources, submit.url)
            args.append(arg)

    if len(args) > 0:
        with Database() as db:
            sql = """
                INSERT INTO mention (date, symbol_id, mentions,
                                     source, url)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """
            db.executemany(sql, args)
            db.commit()


def show_reddit(subsources, num_days):
    with Database() as db:
        query = """
            SELECT COUNT(*) AS num_mentions, symbol
            FROM mention JOIN symbols ON symbols.id = mention.symbol_id
            WHERE date(date) > current_date - interval '%s day'
            GROUP BY symbol_id, symbol   
            HAVING COUNT(symbol) > 10
            ORDER BY num_mentions DESC
        """
        counts = db.query(query, (num_days,))

        query = """
            SELECT symbol, mentions, url, date, source
            FROM mention JOIN symbols ON symbols.id = mention.symbol_id
            WHERE source = %s
            ORDER BY date DESC
            LIMIT 50;
        """
        mentions = db.query(query, (subsources,))

        rows = db.fetchall()

    return counts, mentions, rows


def _show_reddit_dict():
    analysis = {}
    with Database() as db:
        query = "SELECT COUNT(*) FROM mention"
        analysis['count'] = db.query(query)

        query = """
            SELECT COUNT(*) AS num_mentions, symbol_id, symbol
            FROM mention JOIN symbols ON symbols.id = mention.symbol_id
            GROUP BY symbol_id, symbol
            HAVING COUNT(*) > 2
            ORDER BY num_mentions DESC;
        """
        analysis['mentions'] = db.query(query)

    for k, v in analysis.items():
        print(k, v)
