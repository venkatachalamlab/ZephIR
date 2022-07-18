# Author: vivekv2@gmail.com

from typing import Callable

def default_args(default_args: str) -> Callable:
    """This returns a decorator that attaches a default_args field to the
    function object, which is used to populate the arg textfield in the app."""

    def dec(fn: Callable) -> Callable:
        fn.default_args = default_args
        return fn

    return dec
