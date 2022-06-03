# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:58:25 2021

@author: Arthur.Gartner
"""
from itertools import accumulate
from requests import session
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from collections import Counter
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
# from tensorflow.keras import losses
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Embedding, LSTM, SimpleRNN, Input
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

nums = 9
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
    


def random_forest(X_train_values, y_train_values, X_test_values, y_test_values, estimators):
    print('~~~RANDOM FOREST~~~')
        
    clf = RandomForestClassifier(n_estimators=estimators)

    clf.fit(X_train_values, y_train_values)
    
    y_pred=clf.predict(X_test_values)

    score = metrics.accuracy_score(y_test_values, y_pred)

    print("Accuracy: " + str(score))
    
    print_confusion_matrix(clf, X_test_values, y_test_values, 'Random Forest')
    
    
def svm_apply(X_train_values, y_train_values, X_test_values, y_test_values):
    print('~~~SVM (kernel= rbf)~~~')
    clf = svm.SVC(kernel='rbf', C = 15)
    clf.fit(X_train_values, y_train_values)
    y_pred=clf.predict(X_test_values)
    score = metrics.accuracy_score(y_test_values, y_pred)
    print("Accuracy: " + str(score)) 
    print_confusion_matrix(clf, X_test_values, y_test_values, 'SVM (rbf)')
    
def mlp_apply(X_train_values, y_train_values, X_test_values, y_test_values):
    print('~~~MLP~~~')
    
    clf = MLPClassifier(solver='lbfgs')
    
    clf.fit(X_train_values, y_train_values)
    
    y_pred=clf.predict(X_test_values)

    score = metrics.accuracy_score(y_test_values, y_pred)

    print("Accuracy: " + str(score))
    
    print_confusion_matrix(clf, X_test_values, y_test_values, 'MLP')

def dnn_apply(X_train_values, y_train_values, X_test_values, y_test_values):
    print('~~~RNN~~~')

    inputs = X_train_values.shape
    print(y_train_values)

    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_trainScaled = scaler.fit_transform(X_train_values)
    X_testScaled = scaler.fit_transform(X_test_values)

    items = Counter(y_train_values).keys()
    print("No of unique items in the list are:", len(items))

    print(y_train_values.shape)
    y_train_values = y_train_values.reshape(len(y_train_values), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(y_train_values)

    y_test_values = y_test_values.reshape(len(y_test_values), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded_test = onehot_encoder.fit_transform(y_test_values)

    print(onehot_encoded)
    half = int((inputs[1]+len(items))/2)
    model = Sequential()

    model.add(Input(shape=(None, inputs[1])))
    # model.add(LSTM(16, input_shape=(None, input), return_sequences=True, stateful=False))
    # model.add(LSTM(8))
    model.add(Dense(inputs[1], activation='sigmoid'))
    # model.add(Dense(half, activation="tanh"))
    model.add(Dropout(0.3))
    # model.add(Dense(half, activation="tanh"))
    model.add(Dense(len(items), activation='softmax'))

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    es = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min')
    mc = ModelCheckpoint('models/saved/best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


    print("\nTraining...")
    model.fit(
        x=X_trainScaled,
        y=onehot_encoded,
        # validation_split = 0.2,
        # validation_data=(data.valid_data, data.valid_labels),
        batch_size=5,
        epochs=200,
        verbose=1,
        callbacks=[mc]
        )

    model = load_model('models/saved/best_model.h5')

    # Predictions against test data set
    predictions = model.predict(
        x=X_testScaled,
        batch_size=1,
        verbose=1
    )

    # Predictions set as 1 or 0
    rounded_predictions = np.argmax(predictions, axis=-1)
    print(rounded_predictions)
    print(onehot_encoded_test)
    y_pred = np.argmax(onehot_encoded_test, axis=-1)
    print(y_pred)

    # Print predictions next to labels
    if len(rounded_predictions) == len(y_pred):
        sum = 0

        for i in range(0, len(rounded_predictions)):
            if rounded_predictions[i] == y_pred[i]:
                sum = sum + 1
        print("\n Sum: ", sum, "Total: ", len(rounded_predictions))
        print("\n Percent: ", (sum/(len(rounded_predictions))) * 100, "%")
    
    cm = confusion_matrix(y_pred, rounded_predictions)


    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 10})
    sn.heatmap(cm/np.sum(cm), annot=True, fmt=".2%", cmap='Blues')
    plt.xlabel("Predictions")
    plt.ylabel("Actual")
    plt.show()


def train_test_model(hparams, inputs, items):

    model = Sequential()

    model.add(Input(shape=(None, inputs[1])))
    # model.add(LSTM(16, input_shape=(None, input), return_sequences=True, stateful=False))
    # model.add(LSTM(8))
    model.add(Dense(int(inputs[1]), activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(hparams[HP_NUM_UNITS], activation=hparams[HP_ACTIVATION_1]))
    model.add(Dropout(hparams[HP_DROPOUT]))
    # model.add(Dense(int(inputs[1]*1.5), activation='tanh'))
    # model.add(Dense(int(inputs[1]*.1), activation='tanh'))
    # model.add(Dense(int(inputs[1]*.5), activation='linear'))
    # model.add(Dense(int(inputs[1]*.25), activation='linear'))
    model.add(Dense(len(items), activation='softmax'))

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    es = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min')
    mc = ModelCheckpoint('models/saved/best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


    # print("\nTraining...")
    # model.fit(
    #     x=X_trainScaled,
    #     y=onehot_encoded,
    #     validation_split = 0.2,
    #     # validation_data=(data.valid_data, data.valid_labels),
    #     batch_size=5,
    #     epochs=100,
    #     verbose=1,
    #     callbacks=[mc]
    #     )

    model.fit(X_trainScaled, onehot_encoded, validation_split = 0.2, epochs=30)
    _, accuracy = model.evaluate(X_testScaled, onehot_encoded_test)

    return accuracy

def run(run_dir, hparams, inputs, items):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_test_model(hparams, inputs, items)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

    
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

X_trainScaled = np.load(X_train_file)
onehot_encoded = np.load(y_train_file)
X_testScaled = np.load(X_test_file)
onehot_encoded_test = np.load(y_test_file)

inputs = X_trainScaled.shape
items = Counter(np.argmax(onehot_encoded, axis=-1)).keys()

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([int(inputs[1]*.1), int(inputs[1]*.25), int(inputs[1]*.5), int(inputs[1]*.75)]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
HP_ACTIVATION_1 = hp.HParam('activ_1', hp.Discrete(['relu', 'softmax', 'tanh', 'sigmoid', 'linear']))
HP_ACTIVATION_2 = hp.HParam('activ_2', hp.Discrete(['relu', 'softmax', 'tanh', 'sigmoid', 'linear']))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))
HP_NUM_LAYERS = hp.HParam('layers', hp.Discrete([1, 2, 3]))

METRIC_ACCURACY = 'val_accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(hparams=[HP_ACTIVATION_1, HP_NUM_UNITS, HP_DROPOUT],
                      metrics=[hp.Metric(METRIC_ACCURACY, display_name='Val ACC')],)

session_num = 1
for num_units in HP_NUM_UNITS.domain.values:
    for activ_1 in HP_ACTIVATION_1.domain.values:
        # for activ_2 in HP_ACTIVATION_2.domain.values:
        # for layer in HP_NUM_LAYERS.domain.values:
        for drop in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            # for lr in HP_LR.domain.values:
            hparams = {HP_ACTIVATION_1: activ_1, HP_NUM_UNITS: num_units, HP_DROPOUT: drop}
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams, inputs, items)
            session_num += 1





        


