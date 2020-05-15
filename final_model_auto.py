# To import the dependent modules.
#from tensorflow.keras.layers import Dense, LSTM, Dropout
#from tensorflow.keras import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os
def Complete_data_modelling(data, train_percent, look_Back, \
                            start_col, end_col, model_filename, optimizer_Name, loss_Name, noofunits, multiple_units,\
                            activation_Function, dropout_Unit, hidden_Layer_Count, Epochs, batch_Size, fig_size_x, fig_size_y, \
                            Y_pred_label, Y_actual_label, graph_Title, x_Label, y_Label,start_time):
  """
  data = Date of type dataframe with all the actuals and to be predicted columns. (data = input_Data)

  train_percent = Percentage of data to be trained, the rest will be the test data [Varies between 0.5 - 0.8]. (train_percent=0.8)

  look_Back = Number of look back days. (look_Back = 30)

  start_col = start index of the coumn from data that is to be predicted. (start_col=0)

  end_col = End index of the coumn from data that is to be predicted. (end_col=1)

  start_col, end_col = [0:1] which means it will predict first column of data.

  model_filename = Name of the file with whcih model needs to be saved in current location. (model_filename = "Nifty_30days")

  optimizer_Name = Name of the Optimizer to be used for Sequeantial model. (optimizer_Name = "adam")

  loss_Name = Name of the loss function used for Sequeantial model. (loss_Name = 'mean_squared_error')

  noofunits = No of units to be defined in LSTM layers. (noofunits = 40)

  multiple_units = No of units added to Units per consecutive layer.(multiple_units = 10)

  activation_Function = Function to be used as activation in Sequential model.(activation_Function = 'relu')

  dropout_Unit = Percentage of units to be dropped after each layer [It varies between 0.2 - 0.8]. (dropout_Unit = 0.2)

  hidden_Layer_Count = Number of hidden layers you want. (hidden_Layer_Count = 3)

  Epochs = No of Epochs to be defined in Sequential model. (Epochs=50)

  batch_Size = Batch Size to be defined in Sequential model [No of samples per batch]. (batch_Size = )

  fig_size_x = Figure size of X axis of the graph. (fig_size_x = 18)

  fig_size_y = Figure size of Y axis of the graph. (fig_size_y = 10)

  Y_pred_label = Label to be seen for Prediction Values. (Y_pred_label = "Nifty Open Predictions")

  Y_actual_label = Label to be seen for Actual Values. (Y_actual_label = "Nifty Open Actuals")

  graph_Title = Title of the Graph. (graph_Title = "Nifty Predictions")

  x_Label = Label to be seen on the X axis. (x_Label = "Nifty Price")
  
  y_Label = Label to be seen on the Y axis. (y_Label = "Date Time")
  """
  data_train = data[0:round(data.shape[0]*train_percent)] 
  data_test = data[round(data.shape[0]*train_percent):] 
  scaler_train = MinMaxScaler()
  X_train, y_train = normalize_divide_chunks(data_train=data_train, scaler=scaler_train, look_back=look_Back, start=start_col, end=end_col)
  model = Sequential()
  model_built =  Model_Design(optimizer_name=optimizer_Name, loss_name=loss_Name, model=model, \
                              units= noofunits, multiple_units=multiple_units, activation_function=activation_Function, \
                              input_shape_row=X_train.shape[1], input_shape_col= X_train.shape[2], \
                              dropout_unit=dropout_Unit, hidden_layer_count=hidden_Layer_Count)
  model_built.summary()
  scaler_test = MinMaxScaler()
  data_new = data_train.tail(look_Back)
  data_new = data_new.append(data_test, ignore_index=True)
  X_test, y_test = normalize_divide_chunks(data_new, scaler=scaler_test, look_back=look_Back, start=start_col, end=end_col)
  history_model = model_built.fit(X_train, y_train, epochs=Epochs, batch_size=batch_Size, validation_data=(X_test, y_test), shuffle=False)
  # predict probabilities for test set
  y_pred = model_built.predict(X_test, verbose=0)
  # predict crisp classes for test set
  y_pred_classes = model_built.predict_classes(X_test, verbose=0)
  # reduce to 1d array
  y_pred = y_pred[:, 0]
  y_pred_classes = y_pred_classes[:, 0]
  #print(y_test,y_pred_classes)
  # accuracy: (tp + tn) / (p + n)
  accuracy = accuracy_score(y_test.round(), y_pred_classes.round())
  print('Accuracy: %f' % accuracy)
  # precision tp / (tp + fp)
  precision = precision_score(y_test.round(), y_pred_classes.round())
  print('Precision: %f' % precision)
  # recall: tp / (tp + fn)
  recall = recall_score(y_test.round(), y_pred_classes.round())
  print('Recall: %f' % recall)
  # f1: 2 tp / (2 tp + fp + fn)
  f1 = f1_score(y_test.round(), y_pred_classes.round())
  print('F1 score: %f' % f1)
  # kappa
  kappa = cohen_kappa_score(y_test.round(), y_pred_classes.round())
  print('Cohens kappa: %f' % kappa)
  # ROC AUC
  auc = roc_auc_score(y_test.round(), y_pred.round())
  print('ROC AUC: %f' % auc)
  # confusion matrix
  results_confusion_matrix = confusion_matrix(y_test.round(), y_pred_classes.round())
  print(results_confusion_matrix)
  #results_confusion_matrix = confusion_matrix(X_act.round(), y_pred.round())
  modelname = "{}_lb{}_noUnit{}_ep{}_bs{}.h5".format(model_filename, str(look_Back), str(noofunits),str(Epochs),str(batch_Size))
  cwd = os.getcwd()
  model_file_path = cwd + "/" + modelname
  print("Model File name: {}".format(model_file_path))
  model_built.save_weights(modelname)
  scale_val = scaler_test.scale_
  y_pred_scaler_val = 1/scale_val[0]
  Y = y_pred * y_pred_scaler_val
  y_act = y_test * y_pred_scaler_val
  Y_list = list(Y)
  Complete_df = data_test
  Complete_df['Prediction'] = Y_list 
  # Visualising the results
  plt.figure(figsize=(fig_size_x, fig_size_y))
  plt.plot(y_act, color = 'red', label = Y_actual_label)
  plt.plot(Y, color = 'blue', label = Y_pred_label)
  plt.title(graph_Title)
  plt.xlabel(x_Label)
  plt.ylabel(y_Label)
  plt.legend(loc='best')
  plt.show()
  end_time = datetime.now()
  exec_time = end_time - start_time
  model_information = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f1': f1, \
                       'Kappa': kappa, 'AUC': auc, 'Confusion Matrix': results_confusion_matrix, 'Model Name': modelname,\
                       'Train Percent': train_percent, 'Look Back': look_Back, 'Optimizer': optimizer_Name, 'Loss': loss_Name,\
                       'No Of Units': noofunits, 'Activation Function':activation_Function, 'Dropout Percent': dropout_Unit,\
                       'Hidden Layer': hidden_Layer_Count, 'Epochs': Epochs, 'Batch Size': batch_Size,'Exection Time': exec_time, 'Actual and Prediction': Complete_df}
  print('Duration: {}'.format(exec_time))
  return model_information


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
  X_train = []
  y_train = []
  pred_column_no = slice(start,end,None)
  for i in range(look_back, data_train.shape[0]):
      X_train.append(data_train[i-look_back:i])
      y_train.append(data_train[i, pred_column_no])
  X_train, y_train = np.array(X_train), np.array(y_train)
  return X_train, y_train



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
    units = units + multiple_units

  model.add(LSTM(units = units, activation = activation_function))
  model.add(Dropout(dropout_unit))

  model.add(Dense(units = 1))
  model.compile(optimizer=optimizer_name, loss = loss_name)
  
  #model.summary()
  
  return model
