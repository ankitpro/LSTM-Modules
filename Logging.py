def Logging_Notebook(notebook_name):

  #importing module 
  import logging, logging.handlers
  import time
  from datetime import datetime
  from pytz import timezone, utc

  def customTime(*args):
      utc_dt = utc.localize(datetime.utcnow())
      my_tz = timezone("Asia/Kolkata")
      converted = utc_dt.astimezone(my_tz)
      return converted.timetuple()

  logger = logging.getLogger('spam_application')
  logging.Formatter.converter = customTime
  #logger.error("customTime")

  ##Creating and Configuring logger
  fhandler = logging.FileHandler(filename=notebook_name, mode='a')
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fhandler.setFormatter(formatter)

  #Setting the threshold of logger to DEBUG 
  logger.addHandler(fhandler)
  logger.setLevel(logging.DEBUG)
  return logger
