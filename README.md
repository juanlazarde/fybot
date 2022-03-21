<img src="https://raw.githubusercontent.com/juanlazarde/fybot/master/fybot/fybot_still.gif" alt="FyBot" width="50%" height="50%">

# FyBot - Financial Bot in Python
Financial dashboard with technical scanner, news, and options analysis.

# Install
Install Python. Latest version 3.9.11. [Download](https://www.python.org/downloads/)

    # linux
    sudo apt update && sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.9

    # win PowerShell
    Register-PackageSource -Name Nuget -Location "http://www.nuget.org/api/v2" –ProviderName Nuget -Trusted
    Install-Package python39 -Scope CurrentUser

Create project directory

    mkdir ~/fybot
    cd fybot

Create a virtual environment

    python -m venv .venv
    ./.venv/Scripts/activate

Windows: First, install Technical Analysis Library (TA-lib): [Download here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

    python -m pip install TA_Lib‑0.4.24‑cpXX‑cpXX‑win_amd64.whl
    # where XX is the Python version

Install dependencies 

    python -m pip install -r requirements.txt

Create Configuration file

    python fybot setup
    
    # verify with linux
    nano ./fybot/config/config.py
    # verify with win
    notepad ./fybot/config/config.py

Install PostgresSQL. [Download here](https://www.postgresql.org/download/). Then create Database tables

    ./.venv/Scripts/activate
    python fybot tables

# Run
Expect a browser to open at `http://localhost:8501` after running this line:

    ./.venv/Scripts/activate
    python fybot

# Docker installation
Install Docker for Windows or Mac. [Download here](https://www.docker.com/products/docker-desktop)

Install Docker for Linux
   
    mkdir fybot
    cd fybot
    curl -LJO https://raw.githubusercontent.com/juanlazarde/fybot/master/docker-install.sh

Download the repository:

    git clone https://github.com/juanlazarde/fybot.git
    cd fybot

Edit `.env` and review `docker-compose.yml`.

Compile and run docker:

    docker compose up -d --build

Create the config file

    docker run -it local/fybot setup  

Check http://localhost:8501

### Fybot Notebooks
* [Reference Library - for collaboration](https://colab.research.google.com/drive/1qHAt9MiIJtdKBuGhlcfL0wNLCAXwo6Pr?usp=sharing)