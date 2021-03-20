import plotly as pt
import plotly.graph_objs as go
import json


class Chart:
    """Chart library"""

    @staticmethod
    def style_candlestick(df):
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])
        fig_json = json.dumps(fig, cls=pt.utils.PlotlyJSONEncoder)
        return fig_json

    @staticmethod
    def finviz():
        pass

    @staticmethod
    def chart(df):
        candlestick = go.Candlestick(x=df['Date'], open=df['Open'],
                                     high=df['High'], low=df['Low'],
                                     close=df['Close'])
        upper_band = go.Scatter(x=df['Date'], y=df['upper_band'],
                                name='Upper Bollinger Band',
                                line={'color': 'red'})
        lower_band = go.Scatter(x=df['Date'], y=df['lower_band'],
                                name='Lower Bollinger Band',
                                line={'color': 'red'})

        upper_keltner = go.Scatter(x=df['Date'], y=df['upper_keltner'],
                                   name='Upper Keltner Channel',
                                   line={'color': 'blue'})
        lower_keltner = go.Scatter(x=df['Date'], y=df['lower_keltner'],
                                   name='Lower Keltner Channel',
                                   line={'color': 'blue'})

        fig = go.Figure(
            data=[candlestick, upper_band, lower_band, upper_keltner,
                  lower_keltner])
        fig.layout.xaxis.type = 'category'
        fig.layout.xaxis.rangeslider.visible = False
        fig.show()
