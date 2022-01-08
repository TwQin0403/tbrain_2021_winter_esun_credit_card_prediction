import logging
import os
from pathlib import Path


class DataLogger:
    def __init__(self):
        self.log_path = Path(os.path.abspath(os.getcwd())) / 'log/data_log'
        logging.basicConfig(filename=self.log_path / "data_log.txt",
                            format='%(asctime)s %(message)s',
                            filemode='a')
        self.logger = logging.getLogger(str(self.log_path))
        self.logger.setLevel(logging.INFO)

    def save_data(self, message):
        self.logger.info(message)
