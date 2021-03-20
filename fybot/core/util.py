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
