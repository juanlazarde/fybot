import time
from functools import wraps
from memory_profiler import memory_usage


def performance(fn):
    """Measure of performance for Time and Memory. NOTE: It runs the
    function twice, once for time, once for memory.

    To measure time for each method we use the built-in *time* module.
    *Perf_counter* provides the clock with the highest available resolution
    To measure *memory* consumption we use the package memory-profiler.

    Use:
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


def fix_path(path_in, path_type='dir'):
    """Fix path, including slashes.

    :path_in: {str} Path to be fixed
    :path_type: {str} 'file' or 'dir'
    """
    path = str(path_in)
    path = path.replace('\\', '/')
    if not path.endswith("/") and path_type != "file":
        path = path + "/"
    return path


class Watchlist:
    """Functions to be used with watchlists and symbols
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
