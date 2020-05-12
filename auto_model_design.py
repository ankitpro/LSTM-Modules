from tensorflow.keras.layers import Dense, LSTM, Dropout
def Model_Design(optimizer_name, loss_name, model, units, multiple_units, activation_function, input_shape_row, input_shape_col, dropout_unit, hidden_layer_count):
  """
  optimizer_name = Name of the optimizer you want to use in Sequential model. (optimiazer_name = 'adam')
  
  loss_name = Name of the loss you want to use in Sequential model. (loss_name ='mean_squared_error')
  
  model = Variable name given to which the model is assigned. (model = Sequential())
  
  units = Dimension of the cell state. (units = 40)
  
  multiple_units = Number of units added to the consecutive layers. (multiple_units=20 [units+20 for each consecutive layer])
  
  activation_function = Name of the activation function used. (activation_function = 'relu')
  
  input_shape_row = Row dimension of your hidden state. (input_shape_row = 1)
  
  input_shape_col = Column dimension of your hidden state. (input_shape_col = 3)
  
  dropout_unit = No of units defined in dropout which varies from 0.2 - 0.8. (dropout_unit = 0.2)
  
  hidden_layer_count = No of hidden layers that you want. (hidden_layer_count = 3)
  """
  model.add(LSTM(units = units, activation = activation_function, return_sequences = True, input_shape = (input_shape_row, input_shape_col)))
  model.add(Dropout(dropout_unit))

  units = units + multiple_units
  for i in range(0, hidden_layer_count):
    model.add(LSTM(units = units, activation = activation_function, return_sequences = True))
    model.add(Dropout(dropout_unit))
    units = units+i*multiple_units

  model.add(LSTM(units = units, activation = activation_function))
  model.add(Dropout(dropout_unit))

  model.add(Dense(units = 1))
  model.compile(optimizer=optimizer_name, loss = loss_name)
  
  model.summary()
  
  return model
