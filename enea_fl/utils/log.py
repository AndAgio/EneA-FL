import os
import sys
import logging


def get_logger(node_type, node_id, log_folder):
    assert node_type in ['server', 'worker', 'federation', 'non_fl']
    if node_type == 'server':
        logger_name = 'server'
    elif node_type == 'worker':
        logger_name = 'worker {}'.format(node_id)
    elif node_type == 'federation':
        logger_name = 'federation'
    else:
        logger_name = 'info'
    my_logger = SmartLogger(logger_name,
                            verbose=False,
                            log_dir=os.path.join(log_folder, node_type))
    if node_type in ['worker', 'server']:
        my_logger.disable_console_output()
    else:
        pass

    def handle_exception(exc_type, exc_value, exc_traceback):
        my_logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return my_logger


class DumbLogger:
    def __init__(self):
        pass

    @staticmethod
    def print_it(msg, *args, **kwargs):
        """Logging method for the EneA_FL level. The `msg` gets logged both to stdout and to file
        (if a file handler is present), irrespective of verbosity settings."""
        return print(msg, *args, **kwargs)

    @staticmethod
    def print_it_same_line(msg, *args, **kwargs):
        """Logging method for the EneA_FL level. The `msg` gets logged both to stdout and to file
        (if a file handler is present), irrespective of verbosity settings."""
        return print(msg, end='\r', *args, **kwargs)

    @staticmethod
    def set_logger_newline():
        print()


class SmartLogger(logging.getLoggerClass()):
    def __init__(self, name, verbose, log_dir='logs'):
        """Create a custom logger with the specified `name`. When `log_dir` is None, a simple
        console logger is created. Otherwise, a file logger is created in addition to the console
        logger.

        This custom logger class adds an extra logging level FRAMEWORK (at INFO priority), with the
        aim of logging messages irrespective of any verbosity settings.

        By default, the five standard logging levels (DEBUG through CRITICAL) only display
        information in the log file if a file handler is added to the logger, but **not** to the
        console.

        :param name: name for the logger
        :param verbose: bool: whether the logging should be verbose; if True, then all messages get
            logged both to stdout and to the log file (if `log_dir` is specified); if False, then
            messages only get logged to the log file (if `log_dir` is specified), with the exception
            of FRAMEWORK level messages which get logged either way
        :param log_dir: str: (optional) the directory for the log file; if not present, no log file
            is created
        """
        # Create custom logger logging all five levels
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        # Add new logging level
        logging.addLevelName(logging.INFO, 'EneA_FL')

        # Determine verbosity settings
        self.verbose = verbose

        # Create stream handler for logging to stdout (log all five levels)
        self.stdout_handler = logging.StreamHandler(sys.stdout)
        self.stdout_handler.setLevel(logging.INFO)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)9s | %(message)s'))
        self.enable_console_output()

        self.file_handler = None
        if log_dir:
            self.log_dir = log_dir
            self.add_file_handler()

    def add_file_handler(self):
        """Add a file handler for this logger with the specified `name` (and store the log file
        under `log_dir`)."""
        # Format for file log
        fmt = '%(asctime)s | %(levelname)9s | %(message)s'
        formatter = logging.Formatter(fmt)

        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
            except:
                print(f'{self.__class__.__name__}: Cannot create directory {self.log_dir}. ',
                      end='', file=sys.stderr)
                self.log_dir = '/tmp' if sys.platform.startswith('linux') else '.'
                print(f'Defaulting to {self.log_dir}.', file=sys.stderr)
        log_file = self.get_log_file()

        # Create file handler for logging to a file (log all five levels)
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(formatter)
        self.addHandler(self.file_handler)

    def get_log_file(self):
        return "{}/{}.log".format(self.log_dir, self.name)

    def has_console_handler(self):
        return len([h for h in self.handlers if type(h) == logging.StreamHandler]) > 0

    def has_file_handler(self):
        return len([h for h in self.handlers if isinstance(h, logging.FileHandler)]) > 0

    def disable_console_output(self):
        if not self.has_console_handler():
            return
        self.removeHandler(self.stdout_handler)

    def enable_console_output(self):
        if self.has_console_handler():
            return
        self.addHandler(self.stdout_handler)

    def disable_file_output(self):
        if not self.has_file_handler():
            return
        self.removeHandler(self.file_handler)

    def enable_file_output(self):
        if self.has_file_handler():
            return
        self.addHandler(self.file_handler)

    def set_logger_inline(self):
        self.stdout_handler.terminator = '\r'

    def set_logger_newline(self):
        self.print_it('\n')
        self.stdout_handler.terminator = '\n'

    def print_it(self, msg, *args, **kwargs):
        """Logging method for the EneA_FL level. The `msg` gets logged both to stdout and to file
        (if a file handler is present), irrespective of verbosity settings."""
        return super().info(msg, *args, **kwargs)

    def print_it_same_line(self, msg, *args, **kwargs):
        """Logging method for the EneA_FL level. The `msg` gets logged both to stdout and to file
        (if a file handler is present), irrespective of verbosity settings."""
        self.set_logger_inline()
        return super().info(msg, *args, **kwargs)

    def _custom_log(self, func, msg, *args, **kwargs):
        """Helper method for logging DEBUG through CRITICAL messages by calling the appropriate
        `func()` from the base class."""
        # Log normally if verbosity is on
        if self.verbose:
            return func(msg, *args, **kwargs)

        # If verbosity is off and there is no file handler, there is nothing left to do
        if not self.has_file_handler():
            return

        # If verbosity is off and a file handler is present, then disable stdout logging, log, and
        # finally reenable stdout logging
        self.disable_console_output()
        func(msg, *args, **kwargs)
        self.enable_console_output()

    def debug(self, msg, *args, **kwargs):
        self._custom_log(super().debug, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._custom_log(super().info, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._custom_log(super().warning, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._custom_log(super().error, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._custom_log(super().critical, msg, *args, **kwargs)