

import streamlit as st

from core.utility import Watchlist
from core.settings import S
import core.option_sniper as osn


@st.cache
def export_results(strategy: str, df: any, path: str):
    """Exports the resulting table.

    Parameters
    ----------
    strategy : str
        Name of the straegy being exported
    df :
        Each strategy has a dataframe.
    path : str
        Path where file will be saved.

    Returns
    -------
    Nothing
    """
    from datetime import datetime
    from core.utility import fix_path

    suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    name = fix_path(f"{path}{strategy}_hacker_{suffix}.csv")
    try:
        df.to_csv(name, index=True, header=True).encode('utf-8')
        st.success(f"Table saved at: {name}")
    except IOError as eid:
        e = RuntimeError("Error saving table. Error: " + str(eid))
        st.exception(e)


def load_profiles():
    """Loads and returns the profiles available.
    # TODO: load profiles in dictionary form from the database

    Returns
    -------
    tuple
        Profile names and corresponding dictionaries.
    """
    profiles = {"Default": S.OPTION_SNIPER,
                "Credit Naked Options": S.OPTION_SNIPER,
                "Credit Vertical Spreads": S.OPTION_SNIPER,
                }
    profile_names = list(profiles.keys())
    return profile_names, profiles


def save_profile(profile_name, params, profiles):
    """Saves profile with selected parameters.
    # TODO: save profiles in dictionary form into the database

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    if profile_name in profiles.keys():
        return False, None, profiles
    profiles[profile_name] = params
    return True


def app():
    # --- SIDEBAR ---
    # Profile to load default values on form.
    profile_names, profiles = load_profiles()
    selected = st.sidebar.selectbox(
        label="Choose a profile",
        options=profile_names,
        index=0,
        help="Populate from profiles"
    )
    params = profiles[selected]

    # Watchlist to load default values on form.
    _watchlist = [', '.join(Watchlist.sanitize(params['WATCHLIST'][k]))
                  for k in params['WATCHLIST']
                  if 'watchlist_' in k]
    if params['WATCHLIST']['watchlist_current'] == '':
        _index = params['WATCHLIST']['selected']
    else:
        _index = list(params['WATCHLIST'].keys()).index('watchlist_current')
    watchlist = st.sidebar.selectbox(
        label="Choose a watchlist",
        options=_watchlist,
        index=_index,
        help="Watchlists saved in the profile"
    )

    # Refresh button
    if st.sidebar.button("Refresh data"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.sidebar.success("Data refreshed")

    # Profile saving form
    with st.sidebar.expander(label="Save profile", expanded=False):
        with st.form(key="profile_name", clear_on_submit=True):
            new_profile_name = st.text_input(
                label="Profile Name",
                help="Name of the new profile. Keep it short"
            )
            submitted = st.form_submit_button("Save")
        if submitted:
            saved = save_profile(new_profile_name, params, profiles)
            if saved:
                st.success(f"Profile {new_profile_name} saved")
            else:
                st.error(f"Profile '*{new_profile_name}*' couldn't be saved,"
                         f"try a different name")

    # Odds and ends/ Debugging
    with st.sidebar.expander(label="Other settings", expanded=False):
        # Export tables and data for futher analysis
        params['DEBUG']['export'] = st.checkbox(
            label="Export data to files?",
            value=params['DEBUG']['export'],
            help="Export tables as files"
        )

        # Force data download
        params['DEBUG']['force_download'] = st.checkbox(
            label="Force download ",
            value=params['DEBUG']['force_download'],
            help="Forces download despite the market being closed. "
                 "Data is unreliable because the market is closed"
        )

    # empty space for mobile compatibilty
    st.sidebar.write("""
    
    """)

    # --- BODY ---
    st.title("Option Sniper 🎯")
    st.write("I'll give you options")

    with st.form(key='my_form'):
        # Symbols to analyze
        params['WATCHLIST']['watchlist_current'] = st.text_input(
            label="Symbols to analyze",
            value=watchlist,
            help="List of stocks to analyze separated by comma"
        )
        params['WATCHLIST']['selected'] = list(params['WATCHLIST'].keys()).index('watchlist_current')

        # Filtering
        with st.expander(label="Symbol filters", expanded=False):
            col1, col2, empty = st.columns(3)
            # Price mimnimum
            params['FILTERS']['min_price'] = col1.number_input(
                label="Price minimum",
                min_value=0.00,
                max_value=1000000.00,
                step=1.00,
                value=float(params['FILTERS']['min_price']),
                help="Analyze symbols with a price greater or equal to this number"
            )

            # Price maximum
            params['FILTERS']['max_price'] = col2.number_input(
                label="Price maximum",
                min_value=1.00,
                max_value=1000000.00,
                step=1.00,
                value=float(params['FILTERS']['max_price']),
                help="Analyze symbols with a price less than or equal to this number"
            )

        # Formating
        col1, col2, col3 = st.columns([1, 1, 1])

        # Premium type: credit or debit
        _index = 0 if params['FILTERS']['premium'] == 'credit' else 1
        params['FILTERS']['premium'] = col1.selectbox(
            label="Premium type",
            options=['credit', 'debit'],
            help="Credit to collect premium or Debit to pay premium",
            index=_index,
        )

        # Strategy: naked, spread, condor, butterfly, etc.
        if 'strategy_selected' not in st.session_state:
            _default = params['FILTERS']['strategies'].split(',')
            _default = [s.strip().lower() for s in _default]
        else:
            _default = st.session_state['strategy_selected']
        strategy = col1.multiselect(
            label="Option Strategy(ies)",
            options=['naked', 'spread'],
            default=_default,
            help="Strategies to be analyzed, i.e. naked, spread, condor",
            key="strategy_selected",
        )
        params['FILTERS']['strategies'] = ','.join(strategy)

        # Option type: put, call
        _option_type = params['FILTERS']['option_type'].split(',')
        _option_type = [s.strip().lower() for s in _option_type]
        _option_type = ",".join(_option_type)

        if _option_type == 'put':
            _index = 0
        elif _option_type == 'call':
            _index = 1
        else:
            _index = 2
        params['FILTERS']['option_type'] = col1.selectbox(
            label="Option type",
            options=['put', 'call', 'put,call'],
            help="Type of option to analyze",
            index=_index,
        )

        # Days to expiration
        params['FILTERS']['max_dte'] = col2.number_input(
            label="Days to Expiration, maximum",
            min_value=0,
            max_value=1000,
            step=1,
            value=int(params['FILTERS']['max_dte']),
            help="Maximum days until option contracts expire"
        )
        params['FILTERS']['min_dte'] = col2.number_input(
            label="Days to Expiration, minimum",
            min_value=0,
            max_value=1000,
            step=1,
            value=int(params['FILTERS']['min_dte']),
            help="Minimum says until option contracts expire"
        )

        # Risk maximum
        params['FILTERS']['max_risk'] = col2.number_input(
            label="Risk maximum",
            min_value=0.00,
            max_value=100000.00,
            step=100.00,
            value=float(params['FILTERS']['max_risk']),
            help="Maximum amount to be exposed to risk with the investment"
        )

        # Expected return on investment
        params['FILTERS']['min_return_pct'] = col2.number_input(
            label="Return minimum",
            min_value=0.00,
            max_value=1000.00,
            step=0.01,
            value=float(params['FILTERS']['min_return_pct']),
            help="Minumum expected return on investment"
        )

        # How wide is the spread?
        if 'spread' in params['FILTERS']['strategies']:
            params['FILTERS']['strike_price_spread'] = col1.number_input(
                label="Strike price spread (for spread strategy)",
                min_value=0.0,
                max_value=5000.0,
                step=0.1,
                value=float(params['FILTERS']['strike_price_spread']),
                help="How wide is the minimum price spread for spread strategy. "
                     "Larger spread menas larger capital required"
            )

        # Delta maximum & minimum
        params['FILTERS']['min_delta'] = col3.number_input(
            label="Delta minimum",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            value=float(params['FILTERS']['min_delta']),
            help="Minimum Delta for the option"
        )
        params['FILTERS']['max_delta'] = col3.number_input(
            label="Delta maximum",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            value=float(params['FILTERS']['max_delta']),
            help="Maximum Delta for the option (1.00 is the max)"
        )

        # Volume, Open Interest, bid/ask percentile
        params['FILTERS']['min_volume'] = col3.number_input(
            label="Volume minimum",
            min_value=0.,
            max_value=100000000.,
            step=1.,
            value=float(params['FILTERS']['min_volume']),
            help="Volume transactions minimum"
        )
        params['FILTERS']['min_open_int'] = col3.number_input(
            label="Open Interest minimum",
            min_value=0.,
            max_value=100.,
            step=1.,
            value=float(params['FILTERS']['min_open_int']),
            help="Open Interest demand minimum"
        )
        params['FILTERS']['max_bid_ask_pct'] = col3.number_input(
            label="Bid/Ask ratio maximum",
            min_value=0.01,
            max_value=100.00,
            step=0.01,
            value=float(params['FILTERS']['max_bid_ask_pct']),
            help="Bid/Ask ratio, maximum allowed in percent (Smaller value means liquid. Ask/Bid-1)"
        )

        # Submit to run Option Sniper script
        # if st.button("Snipe"):
        hunt = st.form_submit_button("Snipe")
        if hunt:
            with st.spinner("Seeking target..."):
                # nested_df is a dictionary, where key is stratey, value is the DF
                nested_df = osn.snipe(params)

                for k in nested_df:
                    if nested_df[k] is not None:
                        df = nested_df[k]
                        df = df.sort_values(by="Return", ascending=False)
                        df = df.style.background_gradient(
                            axis=0,
                            subset=['Return', 'Daily Return', 'IV']
                        )
                        st.dataframe(data=df)

                        if params['DEBUG']['export']:
                            export_results(
                                strategy=k,
                                df=nested_df[k],
                                path=S.DATASET_DIR
                            )
