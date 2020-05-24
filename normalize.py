######################[ OBSELETE ]#################################

import numpy as np
def normalize_divide_chunks(data_train, scaler, look_back, start,end):
  """
  It expects the following data:
  data_train = data to be trained.

  scaler = It expectes MinMaxScaler

  look_back = Number for which the data needs to be divided.

  start = start position of yth column to be predicted.

  end = end position of yth column to be predicted.

  Eg: X_train, y_train = normalize_divide_chunks(X_train, scaler, 30, 0, 3)
  """
  data_train = scaler.fit_transform(data_train)
  print(data_train)
  X_train = []
  y_train = []
  pred_column_no = slice(start,end,None)
  for i in range(look_back, data_train.shape[0]):
      X_train.append(data_train[i-look_back:i])
      y_train.append(data_train[i, pred_column_no])
  X_train, y_train = np.array(X_train), np.array(y_train)
  return X_train, y_train
