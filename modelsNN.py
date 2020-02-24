import tensorflow as tf
#from tensorflow.keras import layers, Model, optimizers


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

        # Parte Conv - LSTM
        
        x2 = tf.keras.layers.LSTM(lstmHiddenSize, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(beta), dropout=0.1)(x1)
        x2 = tf.keras.layers.Dropout(prob)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        
        # parte Cov - 2Conv
        x3 = tf.keras.layers.Conv1D(convBFilter, 2*convAKernel, strides=ConvBstrides, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(x1)
        x3 = tf.keras.layers.Activation('relu')(x3)
        x3 = tf.keras.layers.MaxPooling1D()(x3)
        x3 = tf.keras.layers.Dropout(prob)(x3)
        x3 = tf.keras.layers.Flatten()(x3)


        y = tf.keras.layers.Concatenate(1)([x2,x3])
        y1 = tf.keras.layers.Dense(hidden_units, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu')(y)
        y1 = tf.keras.layers.Dropout(prob)(y1)
        y1 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='sigmoid')(y1)
        # ottimizzazione
        # optim = tf.keras.optimizers.Adam(lr=learn_rate)
        model = tf.keras.models.Model(inputs=[input1], outputs=y1)
        optim = tf.keras.optimizers.Adam(lr=learn_rate)
        if(metrics != None):
            model.compile(optimizer=optim, loss=loss, metrics=metrics) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])
        else:
            model.compile(optimizer=optim, loss=loss) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])

        return model

    #######################################################################################
    ## Modello che combina CONVLSTM e CONV2CONV e FullyConnected
    #######################################################################################
    def Comb3Layer(self, input_size=(151,4), 
                    convAFilter = 50, convAKernel= 5, lstmHiddenSize = 50, stridesConvA=1, #Conv-LSTM Params
                    convBFilter1 = 50, convBKernel1= 3, convBFilter2 = 50, stridesConvB1=2, stridesConvB2=2, #Conv-2Conv Params
                    kDense = 32,
                    prob = 0.5, learn_rate = 0.0003, beta = 1e-3, loss='binary_crossentropy', metrics=None):
        input1 = tf.keras.layers.Input(shape=input_size)
        # Parte Conv - LSTM
        x1 = tf.keras.layers.Conv1D(convAFilter, convAKernel, input_shape=input_size, strides=stridesConvA, kernel_regularizer=tf.keras.regularizers.l2(beta), padding="same")(input1)
        x1 = tf.keras.layers.Activation('relu')(x1)
        x1 = tf.keras.layers.MaxPool1D()(x1)
        x1 = tf.keras.layers.Dropout(prob)(x1)
        x1 = tf.keras.layers.LSTM(lstmHiddenSize, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(beta), dropout=0.1)(x1)
        x1 = tf.keras.layers.Dropout(prob)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        # parte Cov - 2Conv
        x2 = tf.keras.layers.Conv1D(convBFilter1, convBKernel1, input_shape=input_size, strides=stridesConvB1, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(input1)
        x2 = tf.keras.layers.Activation('relu')(x2)    
        x2 = tf.keras.layers.MaxPooling1D()(x2)
        x2 = tf.keras.layers.Dropout(prob)(x2)
            
        x2 = tf.keras.layers.Conv1D(convBFilter2, 2*convBKernel1, strides=stridesConvB2, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(x2)
        x2 = tf.keras.layers.Activation('relu')(x2)
        x2 = tf.keras.layers.MaxPooling1D()(x2)
        x2 = tf.keras.layers.Dropout(prob)(x2)
        x2 = tf.keras.layers.Flatten()(x2)

        # parte Fully connected
        x3 = tf.keras.layers.Dense(kDense, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu')(input1)
        x3 = tf.keras.layers.Flatten()(x3)

        y = tf.keras.layers.Concatenate(1)([x1,x2, x3])
        y1 = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu')(y)
        y1 = tf.keras.layers.Dropout(prob)(y1)
        y1 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='sigmoid')(y1)

        # ottimizzazione
        # optim = tf.keras.optimizers.Adam(lr=learn_rate)
        model = tf.keras.models.Model(inputs=[input1], outputs=y1)
        optim = tf.keras.optimizers.Adam(lr=learn_rate)
        if(metrics != None):
            model.compile(optimizer=optim, loss=loss, metrics=metrics) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])
        else:
            model.compile(optimizer=optim, loss=loss) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])

        return model

    #######################################################################################
    ## Modello che combina diversi livelli convolutivi a stride crescente
    #######################################################################################
    def ConvStrideComb(self, input_size=(151,4), 
                    convAFilter = 50, convAKernel= 5, lstmHiddenSize = 50, stridesConvA=1, #Conv-LSTM Params
                    convBFilter1 = 50, convBKernel1= 3, convBFilter2 = 50, stridesConvB1=1, stridesConvB2=1, #Conv-2Conv Params
                    convCFilter1 = 50, convCKernel1= 3, convCFilter2 = 50, stridesConvC1=2, stridesConvC2=2, #Conv-2Conv Params
                    convDFilter1 = 50, convDKernel1= 3, convDFilter2 = 50, stridesConvD1=4, stridesConvD2=4, #Conv-2Conv Params
                    prob = 0.5, learn_rate = 0.0003, beta = 1e-3, loss='binary_crossentropy', metrics=None):
        input1 = tf.keras.layers.Input(shape=input_size)
        # Parte Conv - LSTM
        x1 = tf.keras.layers.Conv1D(convAFilter, convAKernel, input_shape=input_size, strides=stridesConvA, kernel_regularizer=tf.keras.regularizers.l2(beta), padding="same")(input1)
        x1 = tf.keras.layers.Activation('relu')(x1)
        x1 = tf.keras.layers.MaxPool1D()(x1)
        x1 = tf.keras.layers.Dropout(prob)(x1)
        x1 = tf.keras.layers.LSTM(lstmHiddenSize, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(beta), dropout=0.1)(x1)
        x1 = tf.keras.layers.Dropout(prob)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        # parte Cov - 2Conv stride 1
        x2 = tf.keras.layers.Conv1D(convBFilter1, convBKernel1, input_shape=input_size, strides=stridesConvB1, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(input1)
        x2 = tf.keras.layers.Activation('relu')(x2)    
        x2 = tf.keras.layers.MaxPooling1D()(x2)
        x2 = tf.keras.layers.Dropout(prob)(x2)
            
        x2 = tf.keras.layers.Conv1D(convBFilter2, 2*convBKernel1, strides=stridesConvB2, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(x2)
        x2 = tf.keras.layers.Activation('relu')(x2)
        x2 = tf.keras.layers.MaxPooling1D()(x2)
        x2 = tf.keras.layers.Dropout(prob)(x2)
        x2 = tf.keras.layers.Flatten()(x2)

        # parte Cov - 2Conv stride 2
        x3 = tf.keras.layers.Conv1D(convCFilter1, convCKernel1, input_shape=input_size, strides=stridesConvC1, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(input1)
        x3 = tf.keras.layers.Activation('relu')(x3)    
        x3 = tf.keras.layers.MaxPooling1D()(x3)
        x3 = tf.keras.layers.Dropout(prob)(x3)
            
        x3 = tf.keras.layers.Conv1D(convCFilter2, 2*convCKernel1, strides=stridesConvC2, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(x3)
        x3 = tf.keras.layers.Activation('relu')(x3)
        x3 = tf.keras.layers.MaxPooling1D()(x3)
        x3 = tf.keras.layers.Dropout(prob)(x3)
        x3 = tf.keras.layers.Flatten()(x3)

        # parte Cov - 2Conv stride 4
        x4 = tf.keras.layers.Conv1D(convDFilter1, convDKernel1, input_shape=input_size, strides=stridesConvD1, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(input1)
        x4 = tf.keras.layers.Activation('relu')(x4)    
        x4 = tf.keras.layers.MaxPooling1D()(x4)
        x4 = tf.keras.layers.Dropout(prob)(x4)
            
        x4 = tf.keras.layers.Conv1D(convDFilter2, 2*convDKernel1, strides=stridesConvD2, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(x4)
        x4 = tf.keras.layers.Activation('relu')(x4)
        x4 = tf.keras.layers.MaxPooling1D()(x4)
        x4 = tf.keras.layers.Dropout(prob)(x4)
        x4 = tf.keras.layers.Flatten()(x4)

        y = tf.keras.layers.Concatenate(1)([x1,x2, x3, x4])
        y1 = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu')(y)
        y1 = tf.keras.layers.Dropout(prob)(y1)
        y1 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='sigmoid')(y1)
        # ottimizzazione
        # optim = tf.keras.optimizers.Adam(lr=learn_rate)
        model = tf.keras.models.Model(inputs=[input1], outputs=y1)
        optim = tf.keras.optimizers.Adam(lr=learn_rate)
        if(metrics != None):
            model.compile(optimizer=optim, loss=loss, metrics=metrics) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])
        else:
            model.compile(optimizer=optim, loss=loss) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])

        return model

    #######################################################################################
    ## Modello che combina CONVLSTM e CONV2CONV
    #######################################################################################
    def ConvLSTMComb(self, input_size=(151,4), 
                    convAFilter = 50, convAKernel= 3, lstmHiddenSize = 50, stridesConvA=1, #Conv-LSTM Params
                    convBFilter1 = 50, convBKernel1= 3, convBFilter2 = 50, stridesConvB1=2, stridesConvB2=2, #Conv-2Conv Params
                    prob = 0.5, learn_rate = 0.0003, beta = 1e-3, loss='binary_crossentropy', metrics=None):
        input1 = tf.keras.layers.Input(shape=input_size)
        # Parte Conv - LSTM
        x1 = tf.keras.layers.Conv1D(convAFilter, convAKernel, input_shape=input_size, strides=stridesConvA, kernel_regularizer=tf.keras.regularizers.l2(beta), padding="same")(input1)
        x1 = tf.keras.layers.Activation('relu')(x1)
        x1 = tf.keras.layers.MaxPool1D()(x1)
        x1 = tf.keras.layers.Dropout(prob)(x1)
        x1 = tf.keras.layers.LSTM(lstmHiddenSize, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(beta), dropout=0.1)(x1)
        x1 = tf.keras.layers.Dropout(prob)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        # parte Cov - 2Conv
        x2 = tf.keras.layers.Conv1D(convBFilter1, convBKernel1, input_shape=input_size, strides=stridesConvB1, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(input1)
        x2 = tf.keras.layers.Activation('relu')(x2)    
        x2 = tf.keras.layers.MaxPooling1D()(x2)
        x2 = tf.keras.layers.Dropout(prob)(x2)
            
        x2 = tf.keras.layers.Conv1D(convBFilter2, 2*convBKernel1, strides=stridesConvB2, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same')(x2)
        x2 = tf.keras.layers.Activation('relu')(x2)
        x2 = tf.keras.layers.MaxPooling1D()(x2)
        x2 = tf.keras.layers.Dropout(prob)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        y = tf.keras.layers.Concatenate(1)([x1,x2])
        y1 = tf.keras.layers.Dense(340, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu')(y)
        y1 = tf.keras.layers.Dropout(prob)(y1)
        y1 = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='sigmoid')(y1)
        # ottimizzazione
        # optim = tf.keras.optimizers.Adam(lr=learn_rate)
        model = tf.keras.models.Model(inputs=[input1], outputs=y1)
        optim = tf.keras.optimizers.Adam(lr=learn_rate)
        if(metrics != None):
            model.compile(optimizer=optim, loss=loss, metrics=metrics) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])
        else:
            model.compile(optimizer=optim, loss=loss) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])

        return model

    def Conv_LSTMNN(self, input_size=(151,4), filters = 50, kernel_len = 5, lstm_hidden_size = 50, learn_rate = 0.0003, prob = 0.5, loss='binary_crossentropy', metrics=None):
        beta = 1e-3
        model = tf.keras.Sequential()
        #model.add(tf.keras.layers.InputLayer(input_size, name='input'))
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

    def Conv_2Conv(self, input_size=(151,4), filters1 = 50, kernel_len1 = 3, filters2 = 50, learn_rate = 0.0003, prob = 0.5, loss='binary_crossentropy', metrics=None ):
        """
        Il modello ha un input shape prefissato (151,4) relativo 
        alla rappresentazione delle sequenze adottata.
        
        Modello formato dai seguenti strati:
            - convoluzione di 
                **filters numero** di kernel, di dimensione
                **kernel_len** con **padding=same**, 
                quindi l'uscita e' della stessa dimensione dell'ingresso.
            - attivazione ReLU
            - MaxPooling1D senza parametri
            - Droput con probabilita' 50%
            
            - convoluzione di 
                **filters numero** di kernel, di dimensione
                **kernel_len** con **padding=same**, 
                quindi l'uscita e' della stessa dimensione dell'ingresso.
            - attivazione ReLU
            - MaxPooling1D senza parametri
            - Droput con probabilita' 50%

            
            - Flatten
            
            - strato fully connected (Dense) di 150 unita', kernel_regularized=l2(1e-3),
            attivazione relu
            - Dropout 50%
            - strato fully connected di dimensione 1 con kernel_regularized=l2(1e-3) 
            e attivazione sigmoide
            
        ottimizzazione **Adam** con **learning rate=0.0003** e funzione di loss
        **binary_crossentropy**.
        
        """
        beta = 1e-3
        model =  tf.keras.Sequential()
        
        #model.add(Conv1D(filters=50, kernel_size=3, input_shape=(151, 4), kernel_regularizer=l2(beta), padding='same'))
        #model.add(tf.keras.layers.InputLayer(input_size, name='input'))
        model.add(tf.keras.layers.Conv1D(filters1, kernel_len1, input_shape=input_size, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same'))
        model.add(tf.keras.layers.Activation('relu'))    
        model.add(tf.keras.layers.MaxPooling1D())
        model.add(tf.keras.layers.Dropout(prob))
        
        model.add(tf.keras.layers.Conv1D(filters2, 2*kernel_len1, kernel_regularizer=tf.keras.regularizers.l2(beta), padding='same'))
        model.add(tf.keras.layers.Activation('relu'))    
        model.add(tf.keras.layers.MaxPooling1D())
        model.add(tf.keras.layers.Dropout(prob))
        
        
        model.add(tf.keras.layers.Flatten())    
        #model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))
        model.add(tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='relu'))

        model.add(tf.keras.layers.Dropout(prob))
        model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(beta), activation='sigmoid'))

        optim = tf.keras.optimizers.Adam(lr=learn_rate)
        
        if(metrics != None):
            model.compile(optimizer=optim, loss=loss, metrics=metrics) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])
        else:
            model.compile(optimizer=optim, loss=loss) #[tf.keras.metrics.binary_accuracy, metrics.precision, metrics.recall, metrics.f1score])


        return model