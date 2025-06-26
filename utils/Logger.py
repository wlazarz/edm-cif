"""With this class we can write to the standard output in zipped files.
The output is done with the command logger.info("") or logger.error("")"""
import logging


class Logger:
    def __init__(self, name: str = __name__, level: str = 'INFO'):
        self.printing = self.is_jupyter_notebook()

        self.logger = logging.getLogger(name)
        level = self.get_logger_level_object(level)
        self.logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def info(self, text):
        """Logs an info message, or prints it if in a Jupyter notebook."""
        if self.printing:
            print(str(text))
        else:
            self.logger.info("________________________________________")
            self.logger.info(str(text))
            self.logger.info("________________________________________")

    def get_logger_level_object(self, level):
        """Converts the level name to a logging level object (e.g., 'INFO' to logging.INFO)."""
        return getattr(logging, level.upper(), logging.INFO)

    @staticmethod
    def is_jupyter_notebook():
        """Checks if the code is running inside a Jupyter notebook."""
        try:
            from IPython import get_ipython
            cfg = get_ipython().config
            return cfg['IPKernelApp'] is not None
        except (NameError, ImportError, KeyError, AttributeError):
            return False
