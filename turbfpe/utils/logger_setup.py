import logging
import os
import sys

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"

_USE_COLOR = sys.stderr.isatty() and not os.environ.get("NO_COLOR")

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format=("%(asctime)s %(message)s"), datefmt="%Y-%m-%d %H:%M:%S"
)


def _c(text: str, color: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{color}{text}{_RESET}"


def format_routine_start_log(partn, func_str, data_name) -> str:
    return (
        f"{_c('START', _CYAN)} "
        f"{_c(f'[PART {partn}]', _DIM)} "
        f"{_c(func_str, _YELLOW)} "
        f"{_c('@', _MAGENTA)}{_c(data_name, _MAGENTA)}"
    )


def format_routine_done_log(partn, func_str, data_name, elapsed_time) -> str:
    time_str = (
        f"{elapsed_time:.2f}s" if elapsed_time < 60 else f"{elapsed_time/60:.2f}m"
    )
    return (
        f"{_c('DONE', _GREEN)}  "
        f"{_c(f'[PART {partn}]', _DIM)} "
        f"{_c(func_str, _YELLOW)} "
        f"{_c(f'(in {time_str})', _DIM)} "
        f"{_c('@', _MAGENTA)}{_c(data_name, _MAGENTA)}"
    )
