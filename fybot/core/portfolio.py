import asyncio
import aiohttp
import pandas as pd

from fybot.core.database import Database


class Portfolio:
    def __init__(self):
        self.get_automatic_portfolio()
        # self.create_custom_portfolio()

    def get_automatic_portfolio(self):
        self.ARK()

    class ARK:
        def __init__(self):
            url_base = \
                "https://ark-funds.com/wp-content/fundsiteliterature/csv/"
            files = [
                "ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
                "ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
                "ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
                "ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.csv",
                "ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv",
                "THE_3D_PRINTING_ETF_PRNT_HOLDINGS.csv",
                "ARK_ISRAEL_INNOVATIVE_TECHNOLOGY_ETF_IZRL_HOLDINGS.csv"
            ]
            # funds = [self.get_ark_funds(url_base + file) for file in files]
            # funds = pd.concat(funds, ignore_index=True)
            urls = [url_base + file for file in files]
            funds = asyncio.run(self.get_ark_funds(urls))
            self.save_ark(funds)

        @staticmethod
        async def download(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url=url) as response:
                    return await response.text()

        async def get_ark_funds(self, url):
            dataset = await asyncio.gather(*[self.download(u) for u in url])
            # data = requests.get(url).text
            funds = []
            for data in dataset:
                data = data.replace('"', '')
                data = data.replace('(%)', '').replace('($)', '')
                data = data.splitlines()[:-2]
                data = [i.split(",") for i in data]
                data = pd.DataFrame(data=data[1:], columns=data[0])
                data['date'] = pd.to_datetime(data['date'])
                data = data[data['fund'].str.strip().astype(bool)]
                funds.append(data)
            return pd.concat(funds, ignore_index=True)

        @staticmethod
        def save_ark(data):
            with Database() as db:
                etfs = data['fund'].unique()
                etfs = ", ".join("'{0}'".format(x) for x in etfs)
                query = f"""SELECT id, symbol
                            FROM symbols 
                            WHERE symbol IN ({etfs})"""
                etfs = db.query(query)
                etfs = {i['symbol']: i['id'] for i in etfs}
                for row in data.itertuples():
                    if not row.ticker:
                        continue
                    query = "SELECT id FROM symbols WHERE symbol = %s"
                    holding = db.query(query, (row.ticker,))
                    if not holding:
                        continue
                    query = """
                     INSERT INTO portfolio 
                                 (id, holding_id, date, shares, weight)
                     VALUES (%s, %s, %s, %s, %s)
                     ON CONFLICT DO NOTHING;"""
                    db.execute(query, (etfs[row.fund], holding['id'], row.date,
                                       row.shares, row.weight))
                db.commit()
