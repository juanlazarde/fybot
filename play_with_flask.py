from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return HOME_HTML

HOME_HTML = """
    <html><body>
        <h2>Scan the Market</h2>
        <form action="/greet">
            What's your name? <input type='text' name='username'><br><br>
            consolidating range (percentage) <input type='text' name='consolidating_pct'><br>
            breakout percentage <input type='text' name='breakout_pct'><br>
            ttm squeeze flag <input type='checkbox' name='ttm_squeeze_flag' value = True><br>
            candlestick flag <input type='checkbox' name='candlestick_flag' value = True><br>
            sma filter fast parameter <input type='text' name='sma_filter_fast'><br>
            sma filter slow parameter <input type='text' name='sma_filter_slow'><br>
            <input type='submit' value='Continue'>
        </form>
    </body></html>"""

@app.route('/greet')
def greet():
    username = request.args.get('username', '')

    consolidating_pct = request.args.get('consolidating_pct', '')
    breakout_pct = request.args.get('breakout_pct', '')
    ttm_squeeze_flag = request.args.get('ttm_squeeeze_flag', False)
    candlestick_flag = request.args.get('candlestick_flag', False)
    sma_filter_fast = request.args.get('sma_filter_fast', '')
    sma_filter_slow = request.args.get('sma_filter_slow', '')

    settings = {'consolidating': {'go': True, 'pct': consolidating_pct},
                'breakout': {'go': True, 'pct': breakout_pct},
                'ttm_squeeze': {'go': ttm_squeeze_flag},
                'candlestick': {'go': candlestick_flag},
                'sma_filter': {'go': True, 'fast': sma_filter_fast, 'slow': sma_filter_slow}}

    if consolidating_pct == '':
        settings['consolidating']['go'] = False

    if breakout_pct == '':
        settings['breakout']['go'] = False

    if (sma_filter_fast == '' ) or (sma_filter_slow == ''):
        settings['sma_filter']['go'] = False

    if username == '':
        username = 'World'

    msg = ('consolidating pct is set to ' + str(settings['consolidating']['go']) + ', percent is ' +  settings['consolidating']['pct'] +
           '</br> breakout_pct is set to ' + str(settings['breakout']['go']) + ', percent is ' + settings['breakout']['pct'] +
           '</br> ttm squeeze is set to ' + str(settings['ttm_squeeze']['go']) +
           '</br> candlestick is set to ' + str(settings['candlestick']['go']) +
           '</br> sma fiter is set to ' + str(settings['sma_filter']['go']) +
           '</br> sma fiter settings are - Fast: ' + str(settings['sma_filter']['fast']) + ', Slow: ' + str(settings['sma_filter']['slow']))

    return GREET_HTML.format(username, msg)

GREET_HTML = """
    <html><body>
        <h2>Hello, {0}!</h2>
        {1}
    </body></html>
    """

if __name__ == "__main__":
    # Launch the Flask dev server
    app.run(host="localhost", debug=True)
