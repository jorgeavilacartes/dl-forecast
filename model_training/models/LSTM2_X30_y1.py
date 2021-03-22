from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv1D, 
    MaxPooling1D, 
    Input, 
    Flatten, 
    Dense, 
    BatchNormalization, 
    LeakyReLU, 
    ReLU,
    Dropout,
    concatenate,
    LSTM,
    Lambda
    )

# Reference name of model
MODEL_NAME = str(Path(__file__).resolve().stem)

# Default inputs
# Dictionary with {"Reference-name-of-input": {"len_input": <int>, "leads": <list with name of leads>}}
INPUTS = OrderedDict(
            TS=dict(
                len_input=30,
                input_features=['venta_unidades_dia', 'venta_clp_dia', 'is_promo'],
            )
        )

class ModelTS:
    '''
    Generate an instance of a keras model
    '''
    def __init__(self, n_output: int=1, output_layer: Optional[str]=None,):
        self.inputs = INPUTS
        self.model_name = MODEL_NAME
        self.n_output = n_output
        self.output_layer = output_layer

    # Load model
    def get_model(self,):

        shape_inputs = [(value.get("len_input"), len(value.get("input_features"))) for value in self.inputs.values()]
        
        # Inputs     
        input_ecg = Input(shape=shape_inputs[0])
        
        # LSTM
        x = LSTM(15, activation='tanh',recurrent_activation="sigmoid", kernel_initializer="glorot_uniform",dropout=0.5, recurrent_dropout=0.0)(input_ecg)
        #x = BatchNormalization()(x)
        # ---------------------------- Dense layer ------------------------------------
        output = Flatten()(x)
        output = Dense(32, kernel_initializer='glorot_normal')(output)
        output = Dropout(0.4)(output)
        output = ReLU()(output)
        output = Dense(self.n_output, activation=self.output_layer, dtype = tf.float32)(output)
        model = Model(inputs = input_ecg, outputs = output)

        return model