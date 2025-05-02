import inspect
import io
import sys

from loguru import logger

LOGGER_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level:<8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
_PRIVATE_LOGGER_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level:<8}</level> | "
    "<cyan>{extra[name]}</cyan>:<cyan>{extra[function]}</cyan>:<cyan>{extra[line]}</cyan> - <level>{message}</level>"
)
logger.level("DEBUG", color="")
logger.level("INFO", color="<green>")
logger.remove()
logger.add(sys.stdout, colorize=True, format=LOGGER_FORMAT)
_param_stack = []


def _param_stack_push(sink, fmt=LOGGER_FORMAT):
    logger.remove()
    logger.add(sink, colorize=True, format=fmt)
    _param_stack.append((sink, fmt))


def _param_stack_pop():
    if _param_stack:
        _param_stack.pop()
    if not _param_stack:
        sink, fmt = sys.stdout, LOGGER_FORMAT
    else:
        sink, fmt = _param_stack[-1]
    logger.remove()
    logger.add(sink, colorize=True, format=fmt)


def _format_string(callback, message: str):
    with io.StringIO() as string:
        _param_stack_push(string, _PRIVATE_LOGGER_FORMAT)
        callback(message)
        _param_stack_pop()
        return string.getvalue().rstrip()


def _get_bind(last=1):
    frame = inspect.currentframe()
    try:
        # Get the caller's frame
        for _ in range(last + 1):
            frame = frame.f_back
        name = frame.f_globals["__name__"]
        function = frame.f_code.co_name
        line = frame.f_lineno
        return {"name": name, "function": function, "line": line}
    finally:
        del frame


def _log(logger_callback, sink_callback, message: str):
    with io.StringIO() as string:
        _param_stack_push(string, _PRIVATE_LOGGER_FORMAT)
        logger_callback(message)
        _param_stack_pop()
        formatted_massage = string.getvalue().rstrip()
    if sink_callback is None:
        print(formatted_massage)
    else:
        sink_callback(formatted_massage)


def trace(message: str, callback=None):
    _log(logger.bind(**_get_bind()).trace, callback, message)


def debug(message: str, callback=None):
    _log(logger.bind(**_get_bind()).debug, callback, message)


def info(message: str, callback=None):
    _log(logger.bind(**_get_bind()).info, callback, message)


def success(message: str, callback=None):
    _log(logger.bind(**_get_bind()).success, callback, message)


def warning(message: str, callback=None):
    _log(logger.bind(**_get_bind()).warning, callback, message)


def error(message: str, callback=None):
    _log(logger.bind(**_get_bind()).error, callback, message)


def critical(message: str, callback=None):
    _log(logger.bind(**_get_bind()).critical, callback, message)
