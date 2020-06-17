import tensorflow as tf

class modelsClass:

    #######################################################################################
    ## Modello che combina CONVLSTM e CONV2CONV
    #######################################################################################
    def ConvLSTMCombV2(self, input_size=(151,4), 
                    convAFilter = 50, convAKernel= 5, ConvAstrides=1, #Conv-Init
                    lstmHiddenSize = 50, #LSTM Params
                    convBFilter = 50,  ConvBstrides=1, #2Conv Params
                    hidden_units = 256, #Dense Params
                    prob = 0.5, learn_rate = 0.0003, beta = 1e-3, loss='binary_crossentropy', metrics=None):
        input1 = tf.keras.layers.Input(shape=input_size)
        
        x1 = tf.keras.layers.Conv1D(convAFilter, convAKernel, input_shape=input_size, strides=ConvAstrides, kernel_regularizer=tf.keras.regularizers.l2(beta), padding="same")(input1)
        x1 = tf.keras.layers.Activation('relu')(x1)
        x1 = tf.keras.layers.MaxPool1D()(x1)
        x1 = tf.keras.layers.Dropout(prob)(x1)

        # Conv - LSTM 
        
        x2 = tf.keras.layers.LSTM(lstmHiddenSize, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(beta), dropout=0.1)(x1)
        x2 = tf.keras.layers.Dropout(prob)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        
        # Cov - 2Conv 

        x3 = tf.keras.layers.Conv1D(convBFilter, 2*convAKernel, strides=ConvBstrides, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(x1)
        x3 = tf.keras.layers.Activation('relu')(x3)
        x3 = tf.keras.layers.MaxPooling1D()(x3)
        x3 = tf.keras.layers.Dropout(prob)(x3)
        x3 = tf.keras.layers.Flatten()(x3)

        # Dense final layers

        y = tf.keras.layers.Concatenate(1)([x2,x3])
        y1 = tf.keras.layers.Dense(hidden_units, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu')(y)
        y1 = tf.keras.layers.Dropout(prob)(y1)
        y1 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='sigmoid')(y1)
        
        # Generate Model from input and output
        model = tf.keras.models.Model(inputs=[input1], outputs=y1)
        # Optimizer
        optim = tf.keras.optimizers.Adam(lr=learn_rate)
        # Compile
        if(metrics != None):
            model.compile(optimizer=optim, loss=loss, metrics=metrics) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])
        else:
            model.compile(optimizer=optim, loss=loss) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])

        return model

    #######################################################################################
    ## Modello CONVLSTM
    #######################################################################################

    def Conv_LSTMNN(self, input_size=(151,4), filters = 50, kernel_len = 5, lstm_hidden_size = 50, learn_rate = 0.0003, prob = 0.5, loss='binary_crossentropy', metrics=None):
        beta = 1e-3
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(filters, kernel_len, input_shape=input_size, kernel_regularizer=tf.keras.regularizers.l2(beta), padding="same"))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool1D())
        model.add(tf.keras.layers.Dropout(prob))
        model.add(tf.keras.layers.LSTM(lstm_hidden_size, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(beta), dropout=0.1))
        model.add(tf.keras.layers.Dropout(prob))
        model.add(tf.keras .layers.Flatten())
        model.add(tf.keras.layers.Dense(150,  kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu'))
        model.add(tf.keras.layers.Dropout(prob))
        model.add(tf.keras.layers.Dense(1,  kernel_regularizer=tf.keras.regularizers.l2(beta), activation='sigmoid'))
        
        optim = tf.keras.optimizers.Adam(lr=learn_rate)
        if(metrics != None):
            model.compile(optimizer=optim, loss=loss, metrics=metrics) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])
        else:
            model.compile(optimizer=optim, loss=loss) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])

        return model

    #######################################################################################
    ## Modelllo CONV2CONV
    #######################################################################################
    def Conv_2Conv(self, input_size=(151,4), 
                filters1 = 50, kernel_len1 = 3, 
                filters2 = 50, 
                learn_rate = 0.0003, prob = 0.5, loss='binary_crossentropy', metrics=None):
        
        beta = 1e-3
        model =  tf.keras.Sequential()
        
        model.add(tf.keras.layers.Conv1D(filters1, kernel_len1, input_shape=input_size, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same'))
        model.add(tf.keras.layers.Activation('relu'))    
        model.add(tf.keras.layers.MaxPooling1D())
        model.add(tf.keras.layers.Dropout(prob))
        
        model.add(tf.keras.layers.Conv1D(filters2, 2*kernel_len1, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same'))
        model.add(tf.keras.layers.Activation('relu'))    
        model.add(tf.keras.layers.MaxPooling1D())
        model.add(tf.keras.layers.Dropout(prob))
        
        
        model.add(tf.keras.layers.Flatten())    
        model.add(tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu'))

        model.add(tf.keras.layers.Dropout(prob))
        model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='sigmoid'))

        optim = tf.keras.optimizers.Adam(lr=learn_rate)
        
        if(metrics != None):
            model.compile(optimizer=optim, loss=loss, metrics=metrics) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])
        else:
            model.compile(optimizer=optim, loss=loss) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])


        return model