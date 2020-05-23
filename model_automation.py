from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import os
from datetime import datetime

def LSTM_complete_auto(data, train_percent, optimizer_name, loss_name, units, multiple_units, activation_function, dropout_unit,\
             hidden_layer_count, Epochs, Batch_Size, look_back, prediction_column, Y_actual_label, Y_pred_label, \
             graph_Title, fig_size_x, fig_size_y, x_Label, y_Label, model_filename, start_time, verbose=0):
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import precision_score
  from sklearn.metrics import recall_score
  from sklearn.metrics import f1_score
  from sklearn.metrics import cohen_kappa_score
  from sklearn.metrics import roc_auc_score
  from sklearn.metrics import confusion_matrix
  import os
  from datetime import datetime
  if verbose == 1:
    print("------------- Debug Mode -------------")
  print("Start Time: start_time")
  if verbose == 1:
    print("Data: \n{}".format(data.head(5)))
    print("Train Size: {}".format(train_size))
    print("loss_name: {}".format(loss_name))
    print("units: {}".format(units))
    print("optimizer_name: {}".format(optimizer_name))
    print("multiple_units: {}".format(multiple_units))
    print("activation_function: {}".format(activation_function))
    print("dropout_unit: {}".format(dropout_unit))
    print("hidden_layer_count: {}".format(hidden_layer_count))
    print("Epochs: {}".format(Epochs))
    print("Batch_Size: {}".format(Batch_Size))
    print("look_back: {}".format(look_back))
    print("prediction_column: {}".format(prediction_column))
  data = data.set_index(data.columns[0])
  if verbose == 1:
    print("Data_set_index_date: \n{}".format(data.head(5)))
  train_data = data[0:round(data.shape[0]*train_percent)] 
  test_data = data[round(data.shape[0]*train_percent):] 
  if verbose == 1:
    print("train_data: \n{}".format(train_data.head(5)))
    print("test_data: \n{}".format(test_data.head(5)))
  test_data_copy = test_data.copy()
  test_data_copy = test_data_copy.reset_index()
  if verbose == 1:
    print("test_data_copy: \n{}".format(test_data_copy.head(5)))
  test_data_date = pd.DataFrame()
  test_data_date[test_data_copy.columns[0]] = list(test_data_copy[test_data_copy.columns[0]])
  if verbose == 1:
    print("test_data_date: \n{}".format(test_data_date.head(5)))
  from sklearn.preprocessing import MinMaxScaler
  sc=MinMaxScaler(feature_range=(0,1))
  train_norm=sc.fit_transform(train_data)
  X_train=[]
  y_train=[]
  for i in range(look_back,len(train_data)):
    X_train.append(train_norm[i-look_back:i])
    y_train.append(train_norm[i,prediction_column])
  X_train,y_train=np.array(X_train),np.array(y_train)
  if verbose ==1:
    print("X_train: \n{}".format(X_train[0:2]))
    print("y_train: \n{}".format(y_train[0:2]))
  from keras.models import Sequential
  from keras.layers import Dense,LSTM,Dropout
  model = Sequential()
  model_built =  Model_Design(optimizer_name=optimizer_name, loss_name=loss_name, model=model, \
                              units= units, multiple_units=multiple_units, activation_function=activation_function, \
                              input_shape_row=X_train.shape[1], input_shape_col= X_train.shape[2], \
                              dropout_unit=dropout_unit, hidden_layer_count=hidden_layer_count)
  model_built.summary()
  model_built.fit(X_train,y_train,epochs=Epochs,batch_size=Batch_Size)
  modelname = "{}_lb{}_noUnit{}_ep{}_bs{}.h5".format(model_filename, str(look_back), str(units),str(Epochs),str(Batch_Size))
  cwd = os.getcwd()
  model_file_path = cwd + "/" + modelname
  if verbose ==1:
    print("Model File name: {}".format(model_file_path))
  model_built.save_weights(modelname)
  inputs = train_data.tail(look_back)
  inputs = inputs.copy()
  inputs = inputs.append(test_data)
  inputs=sc.transform(inputs)
  X_test=[]
  y_test=[]
  for i in range(look_back,len(inputs)):
    X_test.append(inputs[i-look_back:i])
    y_test.append(inputs[i,prediction_column])
  X_test, y_test=np.array(X_test), np.array(y_test)
  if verbose ==1:
    print("X_test: \n{}".format(X_test[0:2]))
    print("y_test: \n{}".format(y_test[0:2]))
  pred=model_built.predict(X_test)
  predicted_scaled = pd.DataFrame(data=pred)
  for i in range(1,len(train_data.columns)):
    predicted_scaled[str(i)] = 1
  predicted_scaled = sc.inverse_transform(predicted_scaled)   
  predicted_normalized = pd.DataFrame(data=predicted_scaled)
  Actual_column = test_data.columns[0]
  complete_data = pd.DataFrame()
  complete_data[test_data_date.columns[0]] = list(test_data_date[test_data_date.columns[0]])
  complete_data["Predictions"] = predicted_normalized[predicted_normalized.columns[0]]
  complete_data["Actual"] = list(test_data[test_data.columns[0]])
  complete_data = complete_data.set_index(complete_data.columns[0])
  # predict crisp classes for test set
  y_pred_classes = model_built.predict_classes(X_test, verbose=0)
  # reduce to 1d array
  y_pred = np.array(predicted_scaled)
  y_pred_classes = y_pred_classes[:, 0]
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
  # confusion matrix
  results_confusion_matrix = confusion_matrix(y_test.round(), y_pred_classes.round())
  print('Confusion Matrix: {}'.format(results_confusion_matrix))
  #predicted_normalized = predicted_normalized.drop([2, 3], axis=1)
  import matplotlib.pyplot as plt
  # Visualising the results
  plt.figure(figsize=(fig_size_x, fig_size_y))
  plt.plot(complete_data["Actual"], color = 'blue', label = Y_actual_label)
  plt.plot(complete_data["Predictions"], color = 'red', label = Y_pred_label)
  plt.title(graph_Title)
  plt.xlabel(x_Label)
  plt.ylabel(y_Label)
  plt.legend(loc='best')
  plt.show()
  end_time = datetime.now()
  exec_time = end_time - start_time
  model_information = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f1': f1, \
                        'Kappa': kappa, 'Confusion Matrix': results_confusion_matrix, 'Model Name': modelname,\
                        'Train Percent': train_percent, 'Look Back': look_back, 'Optimizer': optimizer_name, 'Loss': loss_name,\
                        'No Of Units': units, 'Activation Function':activation_function, 'Dropout Percent': dropout_unit,\
                        'Hidden Layer': hidden_layer_count, 'Epochs': Epochs, 'Batch Size': Batch_Size,'Exection Time': exec_time, 'Actual and Prediction': complete_data}
  print('Duration: {}'.format(exec_time))
  return model_information

def Model_Design(optimizer_name, loss_name, model, units, \
                 multiple_units, activation_function, input_shape_row, \
                 input_shape_col, dropout_unit, hidden_layer_count):
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
  from keras.models import Sequential
  from keras.layers import Dense,LSTM,Dropout
  model.add(LSTM(units = units, activation = activation_function, return_sequences = True, input_shape = (input_shape_row, input_shape_col)))
  model.add(Dropout(dropout_unit))

  #units = units + multiple_units
  for i in range(0, hidden_layer_count):
    model.add(LSTM(units = units, activation = activation_function, return_sequences = True))
    model.add(Dropout(dropout_unit))
    #units = units + multiple_units

  model.add(LSTM(units = units, activation = activation_function))
  model.add(Dropout(dropout_unit))

  model.add(Dense(units = 1))
  model.compile(optimizer=optimizer_name, loss = loss_name)
  
  #model.summary()
  
  return model
