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
    
    html.Div(modal1),
    dcc.Dropdown(
                participants_list,
                'P01',
                id='xaxis-column'
    ),
    html.Button(id="button_id", children="Run Job!", n_clicks=1),
    html.Button(id="cancel_button_id", children="Cancel Running Job!", n_clicks=0),
    html.Div(children=[
    html.Div(children=[
    daq.Gauge(
        color={"gradient":True,"ranges":{"red":[0,60],"green":[60,100]}},
        showCurrentValue=True,
        id='my-gauge-1',
        label="Engagement Gauge",
        min=0,
        max=100,
        value=50,
    ),
    ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
    ),

    html.Div(children=[
    html.P(id="paragraph_2", children=["Emotion"]),
    html.Img(id='image',src=app.get_asset_url('happy.jpg'))
    # daq.Indicator(
    #     id='happy-indicator',
    #     label="Happy",
    #     width=60,
    #     height=60,
    #     value=False,
    #     color="green"
    # ),
    # daq.Indicator(
    #     id='sad-indicator',
    #     label="Sad",
    #     width=60,
    #     height=60,
    #     value=False,
    #     color="yellow"
    # ),
    # daq.Indicator(
    #     id='angry-indicator',
    #     label="Angry",
    #     width=60,
    #     height=60,
    #     value=False,
    #     color="red"
    # ),
    # daq.Indicator(
    #     id='calm-indicator',
    #     label="Calm",
    #     width=60,
    #     height=60,
    #     value=False,
    #     color="blue"
    # )],style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}),

    ],style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
    )
])
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
    # Output('happy-indicator', 'value'),
    # Output('sad-indicator', 'value'),
    # Output('angry-indicator', 'value'),
    # Output('calm-indicator', 'value'),
    Output('image', 'src'),
    ],
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
        #path = "F:\TUK\SoSe22\HIWI\dash_display"
 
        for i,l,aro,val in zip(res,labels,aro_labels,val_labels):
            time.sleep(1)
            #print(i,l,aro,val)
            if l==0:
                i=1-i
            if val==1 and aro==1:
                path = 'happy.jpg'
                # H = True
                # A = S = C = False
            elif val==0 and aro==1:
                path = 'angry.jpg'
                # A = True
                # H = S = C = False
            elif val==0 and aro==0:
                path = 'sad.jpg'
                # S = True
                # H = A = C = False
            else:
                path = 'calm.jpg'
                # C = True
                # H = S = A = False
            print(val,aro)
            set_progress((i*100,app.get_asset_url(path)))
        return [part+"Engagement Completed, Select next Participant"]
    #time.sleep(2.0)
    #
    


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