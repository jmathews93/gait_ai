# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:58:25 2021

@author: Arthur.Gartner
"""
#from itertools import accumulate
from requests import session
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from collections import Counter
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
# from tensorflow.keras import losses
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Embedding, LSTM, SimpleRNN, Input, InputLayer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorboard.plugins.hparams import api as hp
import keras_tuner as kt
import os.path



# from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from IPython.display import display, HTML
import seaborn as sns
import math
import decimal

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.neural_network import MLPClassifier
from torch import dropout

nums = 15
#~~~INITIAL CODE SETUP~~~
pd.options.display.float_format = '{:.7f}'.format
decimal.getcontext().prec = 5

#if false then default is left waist
right_pocket = False

gait_segment = True

total_classes = 0

#Get main csv file
data_directory = Path.cwd() / Path('data', 'preprocessed_full', 'FinalMatrix-Gait-Meta-Data - FinalMatrix.csv')

#Assign standard column names
column_names = ["userid", "session", "sensor_location", "samplenumber", "x-axis", "y-axis", "z-axis"]

#Setup blank dataframe
new_df = pd.DataFrame(columns = column_names)

#Initial dataframe read from full csv read for data csv
df = pd.read_csv(data_directory, encoding='utf-8')

#Function to load new_df with correct data in relation to user
def loadNewDF():
    total_users = []
    #Check location flag
    if right_pocket:
        location_string = 'subject_right_pocket'
    else:
        location_string = 'subject_left_waist'
    
    #Iterate through data csv
    for i in range(0, len(df.head(nums))):
        #Get data file location from file location text column
        subject_data1 = Path.cwd() / 'data' / 'preprocessed_full' / 'subjects' / df[location_string + '_session1'].values[i]
        print(subject_data1)
        #Save user id for associated row in data csv
        subject_id = df['subject_number'].values[i]
        #Check if linked subject data file exists
        if (subject_data1.is_file()):
            if subject_id not in total_users:
                total_users.append(subject_id)

            #Read full subject data to dataframe
            subject_read_df = pd.read_csv(subject_data1, delimiter=',', encoding="utf-8-sig")
            #Iterate through data rows
            for n in range(0, len(subject_read_df.head(1000))):
                #Create row from pertanent columns for current row
                row = {'userid' : subject_id, 'session' : '1', 'sensor_location' : location_string, 'samplenumber' : n, 'x-axis' : subject_read_df['motionUserAccelerationX.G.'].values[n],
                       'y-axis' : subject_read_df['motionUserAccelerationY.G.'].values[n], 'z-axis' : subject_read_df['motionUserAccelerationZ.G.'].values[n]}
                global new_df
                #Add row to global datframe
                new_df = new_df.append(row, ignore_index = True)
                
    global total_classes
    total_classes = len(total_users)
    print('Total users: ', len(total_users))
                
#Function to visual data characteristics
def plot_inital_data():
    df['Age'].value_counts().plot(kind='bar',
                                   title='User Ages', color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.rcParams["figure.figsize"] = (10,5)
    plt.show()

    df['Weight'].value_counts().plot(kind='bar',
                                   title='User Weights', color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.show()

    df_row_count = df.count(axis='rows')

    x = [1, 2]

    y = [df_row_count['subject_left_waist_session1'], df_row_count['subject_left_waist_session2']]

    tick_label = ['Session 1', 'Session 2']

    plt.bar(x, y, tick_label=tick_label, width=0.8, color = ['red', 'blue'])

    plt.xlabel('Session number')
    plt.ylabel('Session totals')
    plt.title('Sessions')

    plt.show()
    
#Function to plot x, y, z values for subjects
def plot_example(dataframe):
    fig, (x, y, z) = plt.subplots(nrows=3,
         figsize=(20, 10),
         sharex=True)
    if len(dataframe) > 2:
        saved_row = 0
        for row in range(1, len(dataframe)):
            if dataframe['samplenumber'].values[row] == 0:
                plot_axis(x, new_df['samplenumber'].values[saved_row: row], new_df['x-axis'].values[saved_row: row], 'X-Axis')
                plot_axis(y, new_df['samplenumber'].values[saved_row: row], new_df['y-axis'].values[saved_row: row], 'Y-Axis')
                plot_axis(z, new_df['samplenumber'].values[saved_row: row], new_df['z-axis'].values[saved_row: row], 'Z-Axis')
                saved_row = row
        if row != len(dataframe) - 1:
            plot_axis(x, new_df['samplenumber'].values[saved_row: len(dataframe)], new_df['x-axis'].values[saved_row: len(dataframe)], 'X-Axis')
            plot_axis(y, new_df['samplenumber'].values[saved_row: len(dataframe)], new_df['y-axis'].values[saved_row: len(dataframe)], 'Y-Axis')
            plot_axis(z, new_df['samplenumber'].values[saved_row: len(dataframe)], new_df['z-axis'].values[saved_row: len(dataframe)], 'Z-Axis')

    plt.subplots_adjust(hspace=0.2)
    fig.suptitle('Accelerometer Data Overlay')
    plt.subplots_adjust(top=0.80)
    # TODO
    # plt.show()

#Helper function for plotting x,y,z
def plot_axis(ax, x, y, title):

    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(True)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
    

def segment_data_gait_cycle(dataframe, time_analysis_size, percentage_of_peak):
    print('~~~Cycle Based Segmentation~~~')
    segments = []
    user = []
    i = 0
    #Iterate through samples
    while (i < len(dataframe)):
        if (i + time_analysis_size < len(dataframe)):
            #Find max peak for local segment
            max_z = dataframe['z-axis'].values[i: i + time_analysis_size].max()
            #Set threshold
            threshold = max_z * percentage_of_peak
            local_true_maximum = 0
            n_index = 0
            n_start = i
            for n in range(i, i + time_analysis_size):
                if dataframe['z-axis'].values[n] > 0:
                    if dataframe['z-axis'].values[n] >= local_true_maximum and dataframe['z-axis'].values[n] >= threshold:
                       #Record local maximum (peak)
                       local_true_maximum = dataframe['z-axis'].values[n]
                       #Record index at peak
                       n_index = n
                elif local_true_maximum != 0 and n_start != n_index:
                    x = dataframe['x-axis'].values[n_start: n_index]
                    y = dataframe['y-axis'].values[n_start: n_index]
                    z = dataframe['z-axis'].values[n_start: n_index]
                    
                    print("start: " + str(n_start) + " end: " + str(n_index) + "value: " +str(dataframe['z-axis'].values[n]))
                
                    segments.append([x.tolist(), y.tolist(), z.tolist()])
                
                    userid = stats.mode(dataframe['userid'][n_start:n_index])[0][0]
                    user.append(userid)
                
                    n_start = n_index
                    
                if dataframe['z-axis'].values[n] <= 0:
                    local_true_maximum = 0
            
            if i == n_start:
                i += 1
            else:
                i = n_start
        else:
            #This cancels the loop
            i = len(dataframe)
            
    row_lengths = []
    
    for row in segments:
        for list_item in row:
            row_lengths.append(len(list_item))
    
    max_length = max(row_lengths)          
    
    print('Max peak distance: ',max_length)
    
    print('Total segments found: ', len(segments))
    
    #Pad records with 0
    for row in segments:
        for list_item in row:
            while len(list_item) < max_length:
                list_item.append(0)
                
    #Make shaped array from segment data collection
    reshaped_segments = np.asarray(segments, dtype=np.float64).reshape(len(segments), 3, -1)

    user_array = np.asarray(user)
    print('Label shape: ', user_array.shape)
    print('Data shape: ', reshaped_segments.shape)
    
    return reshaped_segments, user_array

def segment_data_time_cycle(dataframe, time_analysis_size, time_step):
    print('~~~Time Based Segmentation~~~')
    segments = []
    user = []
    i = 1
    while (i < len(dataframe) - time_analysis_size - time_step):
        x = dataframe['x-axis'].values[i: i + time_analysis_size]
        y = dataframe['y-axis'].values[i: i + time_analysis_size]
        z = dataframe['z-axis'].values[i: i + time_analysis_size]

        segments.append([x.tolist(), y.tolist(), z.tolist()])
                
        userid = stats.mode(dataframe['userid'].values[i: i + time_analysis_size])[0][0]
        user.append(userid)
                
        i = i + time_step
        
             
    print('Total segments found: ', len(segments))
    
    #Make shaped array from segment data collection
    reshaped_segments = np.asarray(segments, dtype=np.float64).reshape(len(segments), 3, -1)

    user_array = np.asarray(user)
    print('Label shape: ', user_array.shape)
    print('Data shape: ', reshaped_segments.shape)
       
    return reshaped_segments, user_array

def print_confusion_matrix(clf, X_test_values, y_test_values, title):
    plot_confusion_matrix(clf, X_test_values, y_test_values)
    plt.rcParams["figure.figsize"] = (5,5)
    plt.grid(False)
    plt.title(title)
    plt.show()
    

def model_builder(hp):
    hp_activ1 = hp.Choice('activ_1', values=['linear', 'softmax', 'sigmoid', 'relu', 'tanh'])
    # hp_activ2 = hp.Choice('activ_2', values=['linear', 'softmax', 'sigmoid', 'relu', 'tanh'])
    hp_units1 = hp.Choice('units1', values=[inputs[1], int(inputs[1]*1.2), int(inputs[1]*1.3), int(inputs[1]*1.4), int(inputs[1]*1.5), int(inputs[1]*1.6)])
    hp_drop1 = hp.Choice('dropout', values=[0.1, 0.2, 0.3 ])
    hp_drop2 = hp.Choice('dropout', values=[0.1, 0.2, 0.3 ])
    # print(X_trainScaled.shape)
    print(onehot_encoded.shape)
    # print(X_testScaled.shape)
    print(onehot_encoded_test.shape)
    input()

    model = Sequential()

    model.add(InputLayer(input_shape=(None, inputs[1])))
    model.add(Dense(inputs[1], activation='tanh'))
    
    # for i in range(hp.Int('layers', 2, 6)):
        # model.add(Dense(units=hp.Int('units_' + str(i), 50, 100, step=10),
                                        # activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid']))
    model.add(Dropout(hp_drop1))
    model.add(Dense(hp_units1, activation=hp_activ1))
    model.add(Dropout(hp_drop2))

    # model.add(Dense(half, activation="tanh"))
    model.add(Dense(len(items), activation='softmax'))
   
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

    
def normalize_array(list_passed):
    maximum = 0
    
    for i in range(len(list_passed)):
        for n in range(3):
            maximum = list_passed[i][n].max()
            for j in range(len(list_passed[i][n])): 
                if list_passed[i][n][j] > maximum:
                    maximum = list_passed[i][n][j]
                    
    for i in range(len(list_passed)):
        for n in range(3):
            for j in range(len(list_passed[i][n])): 
                if list_passed[i][n][j] != 0 and maximum != 0:
                    list_passed[i][n][j] = decimal.Decimal(list_passed[i][n][j]) / decimal.Decimal(maximum)
    return list_passed        
        
X_train_file = 'train_x.npy'
y_train_file = 'train_y.npy'
X_test_file = 'test_x.npy'
y_test_file = 'test_y.npy'

if not os.path.exists(X_train_file):
    loadNewDF()    
    plot_example(new_df)  

    if gait_segment:
        X, y = segment_data_gait_cycle(new_df, 300, 0.8)
    else:
        X, y = segment_data_time_cycle(new_df, 400, 200)

    print(X.shape)

    y = y.reshape(len(y), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    y = onehot_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    #if not gait_segment:
        #normalize_array(X_train)
        #normalize_array(X_test)

    print('Shape of X_train array: ', X_train.shape)
    print(X_train.shape[0], 'training samples')
    print('Shape of y_train array: ', y_train.shape)
    # input()

    #Need to flatten for model ingestation
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    inputs = X_train.shape
    print(y_train)

    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_trainScaled = scaler.fit_transform(X_train)
    X_testScaled = scaler.fit_transform(X_test)

    print(y_train.shape)
    onehot_encoded = y_train

    items = Counter(np.argmax(onehot_encoded, axis=-1)).keys()
    print("No of unique items in the list are:", len(items))

    onehot_encoded_test = y_test

    print(onehot_encoded)

    np.save(X_train_file, X_trainScaled)
    np.save(y_train_file, onehot_encoded)
    np.save(X_test_file, X_testScaled)
    np.save(y_test_file, onehot_encoded_test)

else:
    X_trainScaled = np.load(X_train_file)
    onehot_encoded = np.load(y_train_file)
    X_testScaled = np.load(X_test_file)
    onehot_encoded_test = np.load(y_test_file)

    inputs = X_trainScaled.shape
    items = Counter(np.argmax(onehot_encoded, axis=-1)).keys()


# KERAS TUNER
tuner = kt.Hyperband(model_builder, 
                     objective='val_accuracy', 
                     max_epochs=50, 
                     factor=3, 
                     directory='my_dir', 
                     project_name='dnn')
                     
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
tuner.search(X_trainScaled, onehot_encoded, epochs=100, validation_split=0.2, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=20)[0]
# print({best_hps})

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_trainScaled, onehot_encoded, epochs=100, verbose=1, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
# print(val_acc_per_epoch)

# bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
print("[INFO] optimal activation 1: {}".format(best_hps.get("activ_1")))
print("[INFO] optimal activation 2: {}".format(best_hps.get("activ_2")))
print("[INFO] optimal number of filters in dense 1 layer: {}".format(best_hps.get("units1")))
# print("[INFO] optimal dropout: {}".format(best_hps.get("dropout")))
print("[INFO] optimal learning rate: {:.4f}".format(best_hps.get("learning_rate")))
# input()


best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(X_trainScaled, onehot_encoded, epochs=best_epoch, verbose=1, validation_split=0.2)

eval_result = hypermodel.evaluate(X_testScaled, onehot_encoded_test)
print("[test loss, test accuracy]:", eval_result)

# dnn_apply(X_train, y_train, X_test, y_test)



        



