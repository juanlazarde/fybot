![Financial Bot in Python](https://raw.githubusercontent.com/juanlazarde/fybot/master/FyBot.gif "Financial Bot in Python")

# FyBot - Financial Bot in Python
Financial dashboard with technical scanner, news, and options analysis.

## Install
* It's recommended to create a virtual environment (i.e. venv) 
* Install dependencies: `pip3 install -r requirements.txt` (developed in Python 3.8)
* If ta-lib gives error: get [download](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and run `pip3 install A_Lib‑0.4.19‑cpXX‑cpXX‑win_amd64.whl`
* [Download](https://www.postgresql.org/download/) and Install PostgreSQL
* Configure `config.py`, under `config` directory. 
* Encrypt your TDA account: run `fybot` and select **Encrypt TDA Account**
* Create new database, i.e. 'source'.
* Create tables in database by running `python app.py create_table`

## Run
* Run `python fybot`. Normal operation will open a browser.
* **Win**: edit path and create shortcut to `fybot.bat` 

### Fybot's Notebooks
* [Reference Library - for collaboration](https://colab.research.google.com/drive/1qHAt9MiIJtdKBuGhlcfL0wNLCAXwo6Pr?usp=sharing)

### Video Tutorials for this repository:
* [Candlestick Pattern Recognition](https://www.youtube.com/watch?v=QGkf2-caXmc)
* [Building a Web-based Technical Screener](https://www.youtube.com/watch?v=OhvQN_yIgCo)
* [Finding Breakouts](https://www.youtube.com/watch?v=exGuyBnhN_8)
