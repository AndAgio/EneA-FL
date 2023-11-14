import pandas as pd
import torch
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def split_df(df, test_size):
  # read data from pickle
  df = df.sample(frac=1).reset_index(drop=True)

  # split data into train and test
  train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["type"])

  # get the list of columns except the type column
  features = list(train_df.columns)
  features.remove("type")

  # encode labels in the same way for train and test
  label_encoder = LabelEncoder()
  train_df["type"] = label_encoder.fit_transform(train_df["type"])
  test_df["type"] = label_encoder.transform(test_df["type"])
  rares_indexes = train_df["type"].value_counts().nsmallest(2).index.values.astype(int)


  # scale data in the same way for train and test
  scaler = MinMaxScaler()
  train_df[features] = scaler.fit_transform(train_df[features])
  test_df[features] = scaler.transform(test_df[features])

  # get all the values from all the columns except the type column
  X_train = train_df[features].values
  y_train = train_df["type"].values

  # get all the values from the type column
  X_test = test_df[features].values
  y_test = test_df["type"].values

  clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
  clf = clf.fit(X_train, y_train)
  model = SelectFromModel(clf, prefit=True)
  X_train = model.transform(X_train)
  X_train = X_train.reshape((-1, 1, X_train.shape[-1]))

  X_test = model.transform(X_test)
  X_test = X_test.reshape((-1, 1, X_test.shape[-1]))

  # y_train = y_train.reshape((y_train.shape[-1], -1))
  # y_test = y_test.reshape((y_test.shape[-1], -1))

  # print("X_TRAiN:100", X_train[:100])

  return X_train, y_train, X_test, y_test, label_encoder, rares_indexes

def process_data(df, test_size):
  
  # split normal data
  print("------------ [N_BaIoT] splitting normal data ------------")
  X_train, y_train, X_test, y_test, label_encoder, _ = split_df(df, test_size)
  
#   print("X_train type", type(X_train))
#   print("X_train.shape", X_train.shape)
#   print("y_train.shape", y_train.shape)
#   print("X_test.shape", X_test.shape)
#   print("y_test.shape", y_test.shape)

  processed_data = {}
  processed_data["n_classes"] = len(label_encoder.classes_)
  processed_data["x_train"] = X_train
  processed_data["y_train"] = y_train
  processed_data["x_test"] = X_test
  processed_data["y_test"] = y_test

  return processed_data

def process_data_final(processed_data, device, batch_size=200):
    x_train, y_train = processed_data["x_train"], processed_data["y_train"]
    x_test, y_test = processed_data["x_test"], processed_data["y_test"]
    
    # y_train = torch.tensor(y_train).long()
    # y_test = torch.tensor(y_test).long()

    # transform to torch tensor - standard
    tensor_x_train = torch.Tensor(x_train)
    tensor_y_train = torch.Tensor(y_train).to(torch.int64)
    tensor_x_test = torch.Tensor(x_test)    
    tensor_y_test = torch.Tensor(y_test).to(torch.int64)

    print("tensor_x_train dtype", tensor_x_train.dtype)
    print("tensor_y_train dtype", tensor_y_train.dtype)
    print("tensor_x_test dtype", tensor_x_test.dtype)
    print("tensor_y_test dtype", tensor_y_test.dtype)


    # move to GPU if device is cuda
    if "cuda" in str(device):
        print("Moving data to GPU")
        tensor_x_train = tensor_x_train.to(device)
        tensor_y_train = tensor_y_train.to(device)
        tensor_x_test = tensor_x_test.to(device)
        tensor_y_test = tensor_y_test.to(device)

    # create datasets for train and test 
    train_dataset = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train)
    test_dataset = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    final_data = {}
    final_data["train_loader"] = train_loader
    final_data["test_loader"] = test_loader

    train_standard = [(data, target) for _,(data, target) in enumerate(final_data["train_loader"])]
    test_standard = [(data, target) for _,(data, target) in enumerate(final_data["test_loader"])]

    return train_standard, test_standard, final_data

def read_nbaiot_data(folder_location, device, batch_size, test_size, is_iot):
    df = pd.read_csv(folder_location+"/df_iot.csv") if is_iot else pd.read_csv(folder_location+"/df.csv")
    processed_data = process_data(df, test_size)
    return process_data_final(processed_data, device, batch_size)

def nbaio_train_single_epoch(trainer, warm_up=False):
    running_loss = 0.
    counter = 0.
    start = time.time()
    predictions = []
    labels_list = []
    for train_batch_x, train_batch_y in trainer.train_data:
        trainer._optimizer.zero_grad()
        pred = trainer.model(train_batch_x)
        pred = torch.squeeze(pred)
        loss = trainer.criterion(pred, train_batch_y)
        loss.backward()
        trainer._optimizer.step()
        loss = loss.item()
        running_loss += loss
        pred_labels = torch.argmax(pred, dim=1)
        predictions += pred_labels.detach().cpu().numpy().tolist()
        labels_list += train_batch_y.detach().cpu().numpy().tolist()
        counter += 1

        if warm_up:
            if counter == 100:
                break
            continue

        metrics = {'loss': running_loss / counter,
                'acc': accuracy_score(np.asarray(labels_list), np.asarray(predictions)),
                'f1': f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')}
        trainer.print_message(index_batch=counter,
                            batch_size=trainer.batch_size,
                            metrics=metrics,
                            mode='train')
        
    trainer.logger.set_logger_newline()
    final_loss = running_loss / counter
    not warm_up and trainer.sample_per_epochs.append(counter)
    
    stop = time.time()
    energy = 0.  # TODO: find how to compute energy here
    comp = 0.  # TODO: find how to compute flops of model
    return final_loss, energy, stop - start, comp

def nbaio_test(trainer):
    predictions = []
    labels_list = []
    trainer.model.eval()
    running_loss = 0.
    counter = 0.
    for test_batch_x, test_batch_y in trainer.test_data:
        pred = trainer.model(test_batch_x)
        pred = torch.squeeze(pred)
        loss = trainer.criterion(pred, test_batch_y)
        loss = loss.item()
        running_loss += loss
        pred_labels = torch.argmax(pred, dim=1)
        predictions += pred_labels.detach().cpu().numpy().tolist()
        labels_list += test_batch_y.detach().cpu().numpy().tolist()
        counter += 1

    f1 = f1_score(np.asarray(labels_list), np.asarray(predictions), average='weighted')
    accuracy = accuracy_score(np.asarray(labels_list), np.asarray(predictions))
    return {
        'loss': running_loss / counter,
        'accuracy': accuracy,
        'f1': f1
    }
