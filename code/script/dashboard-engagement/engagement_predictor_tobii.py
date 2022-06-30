import pandas as pd
import numpy as np
import scipy.stats as stats


def read_data(part):
        data = pd.read_csv('smalldata.csv')
        return data[data['participant'] == part]

def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3
    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        for column in df.columns:
                column = df[column].values[i: i + frame_size]
                frames.append([column])

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)

    return frames

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

def prepare_data(data):
    Fs = 64
    frame_size = Fs*4 # 240
    hop_size = Fs*2 # 120
    df_val = data
    
    df_val.drop(['participant','Engagement','Eng_Class'], axis=1, inplace=True)
    
    for col in df_val.columns:
        df_val[col] = feature_normalize(df_val[col])

    x_val = get_frames(df_val, frame_size, hop_size)

    num_time_periods1, num_sensors1 = x_val.shape[1], x_val.shape[2]
    num_classes1 = 2

    input_shape1 = (num_time_periods1*num_sensors1)
    x_val = x_val.reshape(x_val.shape[0], input_shape1)

    x_val = x_val.astype("float32")

    return x_val