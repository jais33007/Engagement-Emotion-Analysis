#import dash
from dash import Dash, dcc, html, Input, Output
import dash_daq as daq
from dash.long_callback import DiskcacheLongCallbackManager
import pandas as pd
import numpy as np
import scipy.stats as stats
from keras.models import load_model
import time

import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = Dash(__name__,long_callback_manager=long_callback_manager)

participants_list = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 
                'P09', 'P10', 'P11', 'P12', 'P13', 'P15', 'P16', 'P18','P19']

app.layout = html.Div([
    html.P(id="paragraph_id", children=["Select Participant"]),
    dcc.Dropdown(
                participants_list,
                'P01',
                id='xaxis-column'
    ),
    html.Button(id="button_id", children="Run Job!"),
    html.Button(id="cancel_button_id", children="Cancel Running Job!"),
    daq.Gauge(
        color={"gradient":True,"ranges":{"red":[0,60],"green":[60,100]}},
        showCurrentValue=True,
        id='my-gauge-1',
        label="Engagement Gauge",
        min=0,
        max=100,
        value=30
    ),

])

def read_data(part):
        data = pd.read_csv('smalldata.csv')
        return data[data['participant'] == part]

def get_frames(df, frame_size, hop_size, label_name):

    N_FEATURES = 3
    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        for column in df.columns:
            if column!=label_name:
                column = df[column].values[i: i + frame_size]
                frames.append([column])
            else:
            # Retrieve the most often used label in this segment
                label = stats.mode(df[label_name][i: i + frame_size])[0][0]
                labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

def prepare_data(data):
    Fs = 64
    frame_size = Fs*4 # 240
    hop_size = Fs*2 # 120
    df_val = data
    
    df_val.drop(['participant','Engagement'], axis=1, inplace=True)
    
    for col in df_val.columns:
        if col != 'Eng_Class':
            df_val[col] = feature_normalize(df_val[col])

    x_val, y_val = get_frames(df_val, frame_size, hop_size, 'Eng_Class')

    num_time_periods1, num_sensors1 = x_val.shape[1], x_val.shape[2]
    num_classes1 = 2

    input_shape1 = (num_time_periods1*num_sensors1)
    x_val = x_val.reshape(x_val.shape[0], input_shape1)

    x_val = x_val.astype("float32")
    #y_val = y_val.astype("float32")

    #y_val = np_utils.to_categorical(y_val, num_classes1)

    return x_val, y_val

            
@app.long_callback(Output("paragraph_id", "children"),
    Input('xaxis-column', 'value'),
    Input("button_id", "n_clicks"),
    running=[
        (Output("button_id", "disabled"), True, False),
        (Output("cancel_button_id", "disabled"), False, True),
    ],
    cancel=[Input("cancel_button_id", "n_clicks")],
    progress=Output('my-gauge-1', 'value'),
    manager=long_callback_manager,)

def detect_engagement(set_progress,part,runjob):
    df = read_data(part)
    print("data read")
    #print(df.head())
    x_val, y_val = prepare_data(df)
    print("data prepared")
    #print(x_val[0])

    model = load_model('best_model.h5')
    print("model loaded")
    preds = model.predict(x_val)
    print("res predicted")
    labels = np.argmax(preds,axis=1)
    res = np.max(preds, axis = 1)  
    for i,l in zip(res,labels):
        time.sleep(1)
        print(i,l)
        if l==0:
            i=1-i
        set_progress(i*100)
    return ["Select Participant"]
    #time.sleep(2.0)


if __name__ == '__main__':
    app.run_server(debug=True)