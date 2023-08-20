import logging



def get_logger(name, log_file_path='./logs/temp.log', logging_level=logging.DEBUG, log_format='%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s'):

	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger(name)

	# print(logger.level)
	# print(logger.getEffectiveLevel(),"start")
	
	logger.setLevel(logging_level)
	formatter = logging.Formatter(log_format)
	
	
	file_handler = logging.FileHandler(log_file_path, mode='w')
	file_handler.setLevel(logging_level)
	file_handler.setFormatter(formatter)

	

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging_level)
	stream_handler.setFormatter(formatter)

	
	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	print(file_handler.level,"file")
	print(stream_handler.level,"stream")

	# logger.setLevel(logging_level)
	print(logger.getEffectiveLevel(),"end")

	return logger



