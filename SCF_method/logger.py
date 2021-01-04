import logging

LOG_FORMAT = '{asctime} [{levelname}] [{filename}:{lineno}] {message}'

SCF_logger = logging.getLogger('SCF Calculation')
formatter = logging.Formatter(LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S', style='{')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
SCF_logger.addHandler(stream_handler)
SCF_logger.setLevel(logging.DEBUG)

