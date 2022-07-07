import functools
import time
from typing_extensions import runtime
import math



class ListNode:
    def __init__(self, value=0, next=None) -> None:
        self.val = value
        self.next = next

# def do_twice(func):
#     @functools.wraps(func) # keep itself
#     def wrapper_do_twice(*args, **kwargs):
#         func(*args, **kwargs)
#         return func(*args, **kwargs)
#     return wrapper_do_twice

# @do_twice
# def see(str):
#     return str


# def timer(func):
#     """
#     print the runtime of the decorator
    
#     """
#     @functools.wraps(func) # keep the original name
#     def wrapper_timer(*args, **kwargs):
#         start_time = time.perf_counter()
#         value = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         run_time = end_time - start_time
#         print(f"Finished {func.__name__!r} in {run_time:4f} secs")
#         return value
#     return wrapper_timer

# @timer
# def waste(num_time):
#     for _ in range(num_time):
#         sum([i**2 for i in range(10000)])

# waste(100)

# !r - convert the value to a string using repr().
def debug(func):
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ','.join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__!r}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug

@debug
def make_greeting(name, age=None):
    if age == None:
        return f"Howdy {name}"
    else:
        return f"Howdy {name}! {age}"

make_greeting(name='Yile', age=16)
math.factorial = debug(math.factorial)
def approximate_e(terms=18):
    return sum(1/math.factorial(n) for n in range(terms))

approximate_e(terms=12)
