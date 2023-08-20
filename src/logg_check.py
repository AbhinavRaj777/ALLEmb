
import logging 
from check_logging import get_logger




logger = get_logger(name="testing", log_file_path="./log_test.txt", logging_level=logging.DEBUG)

# logger.setLevel(logging.NOTSET)
print(logger.getEffectiveLevel(),"just before")

logger.debug('Created Relevant Directories')
logger.info('Created Relevant Directories')
logger.error('Created Relevant Directories')
logger.warning('Created Relevant Directories')
logger.critical('Created Relevant Directories')
logging.enable()
print(logger.isEnabledFor(logging.ERROR))
print(logger.getEffectiveLevel(),"just after effect")
print(logger.level,"just after ")