import pandas as pd
import yfinance as yf
import bisect
import datetime
import csv


def transform_date_string(date_str):
    date_str = str(date_str)
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


news = pd.read_csv('data/input/news_reuters.csv', header=None,
                   names=["ticker", "company", "date", "title", "contents", "type"])
tickers = pd.read_csv('data/input/tickerList.csv', header=None, names=["ticker", "name", "type", "value"])

start_date = transform_date_string(min(news["date"]))
start_date = (datetime.datetime.strptime(start_date, "%Y-%m-%d") + datetime.timedelta(days=-10)).strftime("%Y-%m-%d")

end_date = transform_date_string(max(news["date"]))
end_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=30)).strftime("%Y-%m-%d")

fin_data = {}
for ticker in tickers["ticker"]:
    fin_data[ticker] = yf.download('GOOGL', start=start_date, end=end_date)

with open('data/labeledData.csv', 'w', encoding='utf8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ticker", "title", "date", "label"])
    for row in news.itertuples(index=False):
        cur_date = datetime.datetime.strptime(transform_date_string(row.date), "%Y-%m-%d")
        ticker = row.ticker
        title = row.title
        data = fin_data[ticker]
        idx = bisect.bisect_left(data.index, cur_date)
        if cur_date != data.index[idx]:
            idx -= 1
        if data["Adj Close"][idx] < data["Adj Close"][idx + 1]:
            writer.writerow([ticker, title, cur_date.strftime("%Y-%m-%d"), 1])
        else:
            writer.writerow([ticker, title, cur_date.strftime("%Y-%m-%d"), 0])
