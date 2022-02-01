<img src="https://github.com/juanlazarde/fybot/blob/master/fybot/FyBOT.gif?raw=true" alt="FyBot" width="50%" height="50%">

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
* Create tables in database by running `python fybot create_table`

## Run
* Run `python fybot`. Normal operation will open a browser.
* If 'run fybot' does not work, a bat script can be used. In the meantime, use `run streamlit run fybot/app.py`
* **Win**: edit path and create shortcut to `fybot.bat`

### Fybot's Notebooks
* [Reference Library - for collaboration](https://colab.research.google.com/drive/1qHAt9MiIJtdKBuGhlcfL0wNLCAXwo6Pr?usp=sharing)
