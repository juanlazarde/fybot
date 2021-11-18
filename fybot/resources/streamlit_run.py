# from streamlit import bootstrap

import streamlit.bootstrap as bt

real_script = '__main__.py'

bt.run(real_script, f'run.py {real_script}', [], {})
