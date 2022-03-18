import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
import ta
import ta.trend
import talib
import yfinance as yf


def app():
    yf.pdr_override()

    st.title('Technical Analysis Web Application')
    st.write("""
    Shown below are the **Moving Average Crossovers**, **Bollinger Bands**, 
    **MACD's**, **Commodity Channel Indexes**, **Relative Strength Indexes** 
    and **Extended Market Calculators** of any stock!
    """)

    st.sidebar.header('User Input Parameters')

    symbol = st.sidebar.text_input("Ticker", 'AAPL')
    start = st.sidebar.date_input(
        label="Start Date",
        value=datetime.date(2018, 1, 1))
    end = st.sidebar.date_input(
        label="End Date",
        value=datetime.date.today())

    symbol = symbol.upper()
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # Read data
    try:
        data = yf.download(symbol, start, end)
    except Exception as e:
        st.error(f"Error downloading symbol data.\n{e}")
        return

    # Adjusted Close Price
    st.header(f"""
              Adjusted Close Price\n {symbol}
              """)
    st.line_chart(data['Adj Close'])

    # ## SMA and EMA
    # Simple Moving Average
    data['SMA'] = talib.SMA(
        data['Adj Close'],
        timeperiod=20)

    # Exponential Moving Average
    data['EMA'] = talib.EMA(
        data['Adj Close'],
        timeperiod=20)

    # Plot
    st.header(f"""
              Simple Moving Average vs. Exponential Moving Average\n {symbol}
              """)
    st.line_chart(
        data[[
            'Adj Close',
            'SMA',
            'EMA'
        ]])

    # Bollinger Bands
    data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(
        data['Adj Close'],
        timeperiod=20)

    # Plot
    st.header(f"""
              Bollinger Bands\n {symbol}
              """)
    st.line_chart(
        data[[
            'Adj Close',
            'upper_band',
            'middle_band',
            'lower_band'
        ]])

    # ## MACD (Moving Average Convergence Divergence)
    # MACD
    data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(
        data['Adj Close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9)

    # Plot
    st.header(f"""
              Moving Average Convergence Divergence\n {symbol}
              """)
    st.line_chart(
        data[[
            'macd',
            'macdsignal'
        ]])

    # ## CCI (Commodity Channel Index)
    # CCI
    cci = ta.trend.cci(
        data['High'],
        data['Low'],
        data['Close'],
        window=31,
        constant=0.015)

    # Plot
    st.header(f"""
              Commodity Channel Index\n {symbol}
              """)
    st.line_chart(cci)

    # ## RSI (Relative Strength Index)
    # RSI
    data['RSI'] = talib.RSI(
        data['Adj Close'],
        timeperiod=14)

    # Plot
    st.header(f"""
              Relative Strength Index\n {symbol}
              """)
    st.line_chart(data['RSI'])

    # ## OBV (On Balance Volume)
    # OBV
    data['OBV'] = talib.OBV(
        data['Adj Close'],
        data['Volume']) / 10 ** 6

    # Plot
    st.header(f"""
              On Balance Volume\n {symbol}
              """)
    st.line_chart(data['OBV'])

    # Extended Market
    fig, ax1 = plt.subplots()

    # Asks for stock ticker
    sma = 50
    limit = 10

    # calculates sma and creates a column in the dataframe
    data['SMA' + str(sma)] = data.iloc[:, 4].rolling(window=sma).mean()
    data['PC'] = ((data["Adj Close"] / data['SMA' + str(sma)]) - 1) * 100

    mean = round(data["PC"].mean(), 2)
    stdev = round(data["PC"].std(), 2)
    current = round(data["PC"][-1], 2)
    yday = round(data["PC"][-2], 2)

    stats = [
        ['Mean', mean],
        ['Standard Deviation', stdev],
        ['Current', current],
        ['Yesterday', yday]
    ]

    frame = pd.DataFrame(
        stats,
        columns=['Statistic', 'Value'])

    st.header(f"""
              Extended Market Calculator\n {symbol}
              """)
    st.dataframe(frame.style.hide(axis='index'))

    # fixed bin size
    bins = np.arange(-100, 100, 1)
    mpl.rcParams['figure.figsize'] = (15, 10)
    plt.xlim([data["PC"].min() - 5, data["PC"].max() + 5])

    plt.hist(data["PC"], bins=bins, alpha=0.5)
    plt.title(symbol + "-- % From " + str(sma) + " SMA Histogram since " + str(
        start.year))
    plt.xlabel('Percent from ' + str(sma) + ' SMA (bin size = 1)')
    plt.ylabel('Count')

    plt.axvline(x=mean, ymin=0, ymax=1, color='k', linestyle='--')
    plt.axvline(x=stdev + mean, ymin=0, ymax=1, color='gray', alpha=1,
                linestyle='--')
    plt.axvline(x=2 * stdev + mean, ymin=0, ymax=1, color='gray', alpha=.75,
                linestyle='--')
    plt.axvline(x=3 * stdev + mean, ymin=0, ymax=1, color='gray', alpha=.5,
                linestyle='--')
    plt.axvline(x=-stdev + mean, ymin=0, ymax=1, color='gray', alpha=1,
                linestyle='--')
    plt.axvline(x=-2 * stdev + mean, ymin=0, ymax=1, color='gray', alpha=.75,
                linestyle='--')
    plt.axvline(x=-3 * stdev + mean, ymin=0, ymax=1, color='gray', alpha=.5,
                linestyle='--')

    plt.axvline(x=current, ymin=0, ymax=1, color='r', label='today')
    plt.axvline(x=yday, ymin=0, ymax=1, color='blue', label='yesterday')

    # add more x axis labels
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(14))

    st.pyplot(fig)

    # Create Plots
    fig2, ax2 = plt.subplots()

    data = data[-150:]

    data['PC'].plot(label='close', color='k')
    plt.title(symbol + "-- % From " + str(sma) + " SMA Over last 100 days")
    plt.xlabel('Date')
    plt.ylabel('Percent from ' + str(sma) + ' EMA')

    # add more x axis labels
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(8))
    plt.axhline(y=limit, xmin=0, xmax=1, color='r')
    mpl.rcParams['figure.figsize'] = (15, 10)
    st.pyplot(fig2)
