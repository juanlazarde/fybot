import time
from functools import wraps
from memory_profiler import memory_usage
from line_profiler.line_profiler import LineProfiler
import pandas as pd


def performance(fn):
    """
    Measure of performance for Time and Memory. NOTE: It runs the
    function twice, once for time, once for memory.

    To measure time for each method we use the built-in *time* module.
    *Perf_counter* provides the clock with the highest available resolution
    To measure *memory* consumption we use the package memory-profiler.


    Use:
    ::
        from util import performance

        @util.performance
        def some_function():
            x += 1
            print(x)
    """
    @wraps(fn)
    def inner(*args, **kwargs):
        fn_kwargs_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        print(f'\n{fn.__name__}({fn_kwargs_str})')

        # Measure time
        perf_time = time.perf_counter()
        # retval = fn(*args, **kwargs)
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - perf_time
        print(f'Time   {elapsed:0.4}')

        # Measure memory
        mem, retval = memory_usage((fn, args, kwargs),
                                   retval=True,
                                   timeout=200,
                                   interval=1e-7)

        print(f'Memory {max(mem) - min(mem)}')
        return retval
    return inner


def fix_path(path_in: str, path_type='dir') -> str:
    """
    Fix path, including slashes.

    :param path_in: Path to be fixed.
    :param path_type: 'file' or 'dir'
    :return: Path with forward slashes.
    """
    path = str(path_in).replace('\\', '/')
    if not path.endswith("/") and path_type != "file":
        path += "/"
    return path


def timeit(method):
    """
    Decorator to measure Time. Like %timeit.

    To measure time for each method we use the built-in *time* module.

    Use:
    ::
        from util import timeit

        @timeit
        def some_function():
            x += 1
            print(x)
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def line_profile(func):
    """
    Line Profile decorator shows time usage per line.

    Use:
    ::
        from utils import line_profile

        @line_profile
        def function_to_profile(numbers):
            do_something = numbers

    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            prof.print_stats()

    return wrapper


def get_pd_memory_usage(df: pd.DataFrame or pd.Series) -> str:
    """
    Return the memory usage of a pandas object.

    :param df: DataFrame or Series.
    :return: String with MB used.
    """
    if isinstance(df, pd.DataFrame):  # It's a DataFrame
        usage_b = df.memory_usage(deep=True).sum()
    elif isinstance(df, pd.Series):  # It's a Series
        usage_b = df.memory_usage(deep=True)
    else:
        raise TypeError("Needs to be a Panda DataFrame or Series")
    # Transfer Byte to MByte
    usage_mb = usage_b / 1024 ** 2
    return f"{usage_mb:03.3f} MB"


def optimize_pd(
        df_in: pd.DataFrame or pd.Series,
        deal_with_na: str = '',
        verbose: bool = False) -> pd.DataFrame or pd.Series:
    """
    Optimizes DataFrames and Series to decrease memory usage.

        * Changes dtypes to optimal, changes float series w/o decimals to int.
        * Changes object types to category type if present in  less than 50%.
        * Shows the impact to memory usage.
        * (optional) Removes NaN rows or fills these with 0.

    :param df_in: Pandas DataFrame or Series.
    :param deal_with_na: 'fill' to fill missing values (NaN), 'drop' to drop.
    :param verbose: Prints the changes in memory usage.
    :return: Optimized DataFrame or Series.
    """
    # Initialize.
    df = df_in.copy()
    i_was_series = False
    if isinstance(df, pd.Series):
        df = df.to_frame()
        i_was_series = True
    elif isinstance(df, pd.DataFrame):
        pass
    else:
        print(f"The dataset should either be a Dataframe or Series, "
              f"not {df.__name__}")
        return None

    # Optional arguments.
    if deal_with_na == 'fill':
        df.fillna(0, inplace=True)  # fill all missing values with 0.
    elif deal_with_na == 'drop':
        df.dropna(axis=0, how='any', inplace=True)
    else:
        pass

    if verbose:
        print('Original Memory Usage: ', get_pd_memory_usage(df))

    # Make sure that 'float' columns have decimals, otherwise they're integers.
    for c in df.select_dtypes(include=['float']).dtypes.index:
        if df[c].where(df[c] - round(df[c], 0) != 0).sum() == 0:
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast='integer')

    # Changes 'int64' to 'unsigned' if positive-only or 'integer'
    int_ser = df.select_dtypes(include=['integer']).dtypes
    if int_ser.empty is False:
        int_i = int_ser.index
        dow = 'unsigned' if df[df[int_i] < 0].isna().all().all() else 'integer'
        df[int_i] = df[int_i].apply(pd.to_numeric, downcast=dow)
        if verbose:
            print('Optimize -> "integer" type: ', get_pd_memory_usage(df))

    # Changes 'float64' to 'float'
    flt_ser = df.select_dtypes(include=['float']).dtypes
    if flt_ser.empty is False:
        flt_i = flt_ser.index
        df[flt_i] = df[flt_i].apply(pd.to_numeric, downcast='float')
        if verbose:
            print('Optimize -> "float" type: ', get_pd_memory_usage(df))

    # Changes 'object' to 'category' if over 50% of the series have same string
    obj_ser = df.select_dtypes(include=['object']).dtypes
    if obj_ser.empty is False:
        obj_i = obj_ser.index
        for c in obj_i:
            # Skip conversion if there are any numbers in the series.
            if df[c].map(lambda x: isinstance(x, (float, int))).any():
                continue
            # If more than 50% of the series are unique then categorize.
            if len(df[c].unique()) / len(df[c]) < 0.5:
                df.loc[:, c] = df[c].astype('category')
        if verbose:
            print('Optimize -> "object" type: ', get_pd_memory_usage(df))

    # Returns DataFrame or Series, depending on what came in.
    return df.squeeze() if i_was_series else df


class Watchlist:
    """Functions to be used with watchlist and symbols
    Returns the final watchlist to be used.
    """
    @staticmethod
    def sanitize(watchlist):
        """Function to clean up & format watchlist lists

            hello

        :param watchlist: String or list with symbols
        :returns: List with sorted, all-caps strings comma-separated
        """
        tmp = ",".join(watchlist) if type(watchlist) is list else watchlist
        tmp = tmp.upper().strip()
        tmp = tmp.replace(";", " ")
        tmp = tmp.replace(",", " ")
        tmp = tmp.strip('][').split(" ")
        tmp = [string for string in tmp if string.strip() != ""]
        used = set()
        tmp = [x for x in tmp if x not in used and (used.add(x) or True)]
        tmp.sort()
        return tmp

    @staticmethod
    def selected(watchlist: str):
        """Process and select a final watchlist

        :param watchlist: Contains watchlist info.
        :returns: List of symbols. If empty, returns default list.
        """
        watchlist_out = watchlist if len(watchlist) > 0 else ['SPY', 'AMD']
        watchlist_out = Watchlist.sanitize(watchlist_out)

        return watchlist_out
