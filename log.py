import sys
import logging


logger = logging.getLogger("videomatch")


def init_logging(level='debug'):
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'warn':
        logger.setLevel(logging.WARN)
    elif level == 'fatal':
        logger.setLevel(logging.FATAL)
    else:
        raise ValueError("Logging level {} unknown".format(level))

    ch = logging.StreamHandler(sys.stdout)
    stdout_formatter = logging.Formatter('%(levelname)7s [%(filename)15s:%(lineno)03d]:\t%(message)s')
    ch.setFormatter(stdout_formatter)
    logger.addHandler(ch)
