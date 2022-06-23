import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State
import dash_daq as daq
from dash.long_callback import DiskcacheLongCallbackManager
import pandas as pd
import numpy as np
from keras.models import load_model
import engagement_predictor_tobii as eng
import time
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = Dash(__name__,long_callback_manager=long_callback_manager,external_stylesheets=[dbc.themes.BOOTSTRAP])

participants_list = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 
                'P09', 'P10', 'P11', 'P12', 'P13', 'P15', 'P16', 'P18','P19']

modal1 = html.Div(
    [
        #dbc.Button("Open modal", id="open", n_clicks=0),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Message:")),
                dbc.ModalBody(dbc.Label(children=" ",id='message')),
                #dbc.ModalFooter(
                    #dbc.Button(
                       # "Close", id="close", className="ms-auto", n_clicks=0
                    #)
                #),
            ],
            id="modal",
            is_open=False,
        ),
    ]
)


app.layout = html.Div([
    html.P(id="paragraph_1", children=["Engagement"]),
    html.P(id="paragraph_2", children=["Emotion"]),
    html.Div(modal1),
    dcc.Dropdown(
                participants_list,
                'P01',
                id='xaxis-column'
    ),
    html.Button(id="button_id", children="Run Job!", n_clicks=1),
    html.Button(id="cancel_button_id", children="Cancel Running Job!", n_clicks=0),
    daq.Gauge(
        color={"gradient":True,"ranges":{"red":[0,60],"green":[60,100]}},
        showCurrentValue=True,
        id='my-gauge-1',
        label="Engagement Gauge",
        min=0,
        max=100,
        value=50,
    ),

    daq.Thermometer( 
        id ='my-indicator-1', 
        label="Emotion Meter", 
        value=0,
        height=150,
        width=50,
        max=5, 
        min=0, 
        scale={'custom': {
            '1':'Happy','2': 'Angry',
            '3': 'Sad','4': 'Calm',
            }},
        showCurrentValue=False,  
        color='black',
    ),

])



            
@app.long_callback(Output("paragraph_1", "children"),
    Input('xaxis-column', 'value'),
    Input("button_id", "n_clicks"),
    running=[
        (Output("button_id", "disabled"), True, False),
        (Output("cancel_button_id", "disabled"), False, True),
    ],
    cancel=[Input("cancel_button_id", "n_clicks")],
    progress=[Output('my-gauge-1', 'value'),
    Output('my-indicator-1', 'color'),
    Output('my-indicator-1', 'value'),],
    manager=long_callback_manager,)

def detect_engagement(set_progress,part,runjob):
    print(runjob)
    if runjob == 1:
        print(part)
        df = eng.read_data(part)
        print("data read")
        #print(df.head())
        x_val = eng.prepare_data(df)
        print("data prepared")
        #print(x_val[0])

        model = load_model('eng_model.h5')
        model2 = load_model('val_model.h5')
        model3 = load_model('aro_model.h5')
        print("models loaded")

        preds = model.predict(x_val)
        val_preds = model2.predict(x_val)
        aro_preds = model3.predict(x_val)
 
        val_labels = np.argmax(val_preds,axis=1)
        aro_labels = np.argmax(val_preds,axis=1)
        labels = np.argmax(preds,axis=1)
        res = np.max(preds, axis = 1) 
        print("res predicted")
 
        for i,l,aro,val in zip(res,labels,aro_labels,val_labels):
            time.sleep(1)
            #print(i,l,aro,val)
            if l==0:
                i=1-i
            if val==1 and aro==1:
                color = 'red'
                value = 1
            elif val==0 and aro==1:
                color = 'blue'
                value = 2
            elif val==1 and aro==0:
                color = 'green'
                value = 3
            else:
                color = 'yellow'
                value = 4 
            print(aro,val,color)
            set_progress((i*100,color,value))
        return [part+"Engagement Completed, Select next Participant"]
    #time.sleep(2.0)
    #
    
# @app.long_callback(Output("paragraph_2", "children"),
#     Input('xaxis-column', 'value'),
#     Input("button_id", "n_clicks"),
#     running=[
#         (Output("button_id", "disabled"), True, False),
#         (Output("cancel_button_id", "disabled"), False, True),
#     ],
#     cancel=[Input("cancel_button_id", "n_clicks")],
#     progress=Output('my-indicator-1', 'color'),
#     manager=long_callback_manager,)

# def detect_emotion(set_progress,part,runjob):
#     print(runjob)
#     if runjob == 1:
#         print(part)
#         df = emo.read_data(part)
#         print("data read")
#         #print(df.head())
#         x_val, y_val = emo.prepare_data(df)
#         print("data prepared")
#         #print(x_val[0])

#         model2 = load_model('best_val_model.h5')
#         model3 = load_model('best_aro_model.h5')
#         print("models loaded")
#         val_preds = model2.predict(x_val)
#         aro_preds = model3.predict(x_val)
#         print("res predicted")
#         val_labels = np.argmax(val_preds,axis=1)
#         aro_labels = np.argmax(val_preds,axis=1)
#         #res = np.max(preds, axis = 1)  
#         for val,aro in zip(val_labels,aro_labels):
#             time.sleep(1)
#             print(val,aro)
#             if val==1 and aro==1:
#                 set_progress('#DFFF00')
#             elif val==0 and aro==1:
#                 set_progress('#DE3163')
#             elif val==1 and aro==0:
#                 set_progress('#6495ED')
#             else:
#                 set_progress('#9FE2BF')
#         return [part+"Emotion Completed, Select next Participant"]


@app.callback(
    [Output("modal", "is_open"),Output("message", "children")],
    Input('my-gauge-1', 'value'),
    #[State("modal", "is_open")],
)
def toggle_modal(value):
    if type(value) == float and value > 75:
        return [True, "Good work!! Keep it up."]
    elif type(value) == float and value < 50:
        return [True, "Please concentrate!! Read mindfully"]
    else:
        return [False, " "]



if __name__ == '__main__':
    app.run_server(debug=True)