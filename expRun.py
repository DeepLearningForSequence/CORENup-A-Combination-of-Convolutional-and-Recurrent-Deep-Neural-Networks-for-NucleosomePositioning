import argparse
import modelsNN
import evaluator
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score
import math
import json

def createModel(modCode, modName, fold, model, metricList = None):
    #Check if model alredy exists

    if(modCode == 1):
        modelLoaded = model.Conv_2Conv(input_size=input_size, metrics=metricsList)
    elif(modCode == 2): 
        modelLoaded = model.Conv_LSTMNN(input_size=input_size, metrics=metricsList)
    elif(modCode == 3):
        modelLoaded = model.ConvLSTMComb(input_size=input_size, metrics=metricsList)
    elif(modCode == 4):
        modelLoaded = model.ConvStrideComb(input_size=input_size, metrics=metricsList)
    elif(modCode == 5):
        modelLoaded = model.Comb3Layer(input_size=input_size, metrics=metricsList)
    elif(modCode == 6):
        modelLoaded = model.ConvLSTMCombV2(input_size=input_size, metrics=metricsList)
        
    return modelLoaded

parser = argparse.ArgumentParser(description='Nucleosome Classification Experiment')
parser.add_argument('-m', '--models', dest = 'modCodes', choices=[1,2,3,4,5,6], type=int, nargs='+', default=[6],
                    help='NN Models name')
parser.add_argument('-s','--species', dest='specCodes', choices=[1,2,3,4], type=int, nargs='+', default=[1,2,3,4],
                    help='Species Name')
parser.add_argument('-e', '--experiments', dest='exp' , default='Experiment1',
                    help='Experiments Name')
parser.add_argument('-nf', '-nfolds', dest='nFolds', default=20,
                    help='Number of Folds')

args = parser.parse_args()

metricsList = [evaluator.precision, evaluator.recall, evaluator.f1score]
n_fold = args.nFolds
epochs=200
batch_size = 64
shuffle = False #True
seed = None #1001
exp_dir = args.exp

#Number of the species for Roc Curve Figure
fNum = 1

if(os.path.exists("elapsed.json")):
    json.load("elapsed.json")

species=["Elegans", "Melanogaster", "Sapiens", "Yeast"]
modName=["Conv2Conv", "ConvLSTM", "Conv2ConvLSTMComb", "ConvStrideComb", "Comb3Layer", "ConvLSTMCombV2"]
for code in args.specCodes:
    s = species[code-1]
    for mCode in args.modCodes:
        m = modName[mCode-1]
        print(m)
        #Create and set model save dir
        if(not os.path.isdir("./Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/models/"+m)):
            os.makedirs("./Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/models/"+m)
        modelPath = "./Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/models/"+m

        #Load nucleosome and linker and then create dataset and  class labels
        nucList = []
        linkList = []
        with open("./Resource/"+s+"/pickle/nucleosome_encode_"+s+".pickle", "rb") as fp:
            nucList = pickle.load(fp)
        with open("./Resource/"+s+"/pickle/linker_encode_"+s+".pickle", "rb") as fp:
            linkList = pickle.load(fp)

        input_size = (len(nucList[1]), 4)
        labels = np.concatenate((np.ones((len(nucList), 1), dtype=np.float32), np.zeros((len(linkList), 1), dtype=np.float32)), axis=0)
        dataset = np.concatenate((nucList,linkList),0)

        #Create folder to save fold dataset and build kfold
        if(not os.path.exists("./Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/"+str(n_fold)+"fold_"+s+".pickle")):
            folds = evaluator.build_kfold(dataset, labels, k=20, shuffle=shuffle, seed=seed)
            with open("./Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/"+str(n_fold)+"fold_"+s+".pickle", "wb") as fp:
                pickle.dump(folds,fp)
        else:
            with open("./Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/"+str(n_fold)+"fold_"+s+".pickle", "rb") as fp:
                folds = pickle.load(fp)

        #Begin Setting for Roc Curve Evaluation
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig = plt.figure(fNum)
        ax  = fig.add_subplot(111)
        #End Setting for Roc Curve Evaluation

        evaluations = {
                "Accuracy" : [],
                "Precision": [],
                "TPR": [],
                "FPR": [],
                "AUC": [],
                "Sensitivity": [],
                "Specificity": [],
                "MCC":[]
            }

        i = 1
        for fold in folds:

            """
        
            1)Train or Load model
            
            """
            #Check if model alredy exists
            if(not os.path.exists(modelPath+"/"+m+"_bestModel-fold"+str(i)+".hdf5")):

                #If not exist then create the model and fit it with the fold
                modelCallbacks = [
                    tf.keras.callbacks.ModelCheckpoint(modelPath+"/"+m+"_bestModel-fold"+str(i)+".hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
                ]
                modelClass = modelsNN.modelsClass()
                model = createModel(mCode, m, i, modelClass, metricsList)
                print(model.summary())
                if(not os.path.exists(modelPath+"/"+m+".json")):
                    model_json = model.to_json()
                    with open(modelPath+"/"+m+".json", "w") as json_file:
                        json_file.write(model_json)
                model.fit(x=fold["X_train"], y=fold["y_train"], batch_size=batch_size, epochs=epochs, verbose=1, callbacks=modelCallbacks, validation_data=(fold["X_test"], fold["y_test"]), validation_freq=1)
            
            else:
                
                #If exists then load the model and performe evaluation
                print("Modello esistente per fold: "+str(i))
                # load json and create model
                json_file = open(modelPath+"/"+m+".json", 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = tf.keras.models.model_from_json(loaded_model_json)
                model.build(input_shape=input_size)
                #model.summary()
                print(model.summary())
                # load weights into new model
                model.load_weights(modelPath+"/"+m+"_bestModel-fold"+str(i)+".hdf5")

                """
                
                2)Evaluate Model on Fold i

                """

                print("Prevedo le Classi di "+s+" Fold"+str(i))
                y_pred = model.predict(fold["X_test"])
                label_pred = evaluator.pred2label(y_pred)
                # Compute precision, recall, sensitivity, specifity, mcc
                acc = accuracy_score(fold["y_test"], label_pred)
                prec = precision_score(fold["y_test"],label_pred)

                conf = confusion_matrix(fold["y_test"], label_pred)
                if(conf[0][0]+conf[1][0]):
                    sens = float(conf[0][0])/float(conf[0][0]+conf[1][0])
                else:
                    sens = 0.0
                if(conf[1][1]+conf[0][1]):
                    spec = float(conf[1][1])/float(conf[1][1]+conf[0][1])
                else:
                    spec = 0.0
                if((conf[0][0]+conf[0][1])*(conf[0][0]+conf[1][0])*(conf[1][1]+conf[0][1])*(conf[1][1]+conf[1][0])):
                    mcc = (float(conf[0][0])*float(conf[1][1]) - float(conf[1][0])*float(conf[0][1]))/math.sqrt((conf[0][0]+conf[0][1])*(conf[0][0]+conf[1][0])*(conf[1][1]+conf[0][1])*(conf[1][1]+conf[1][0]))
                else:
                    mcc= 0.0
                fpr, tpr, thresholds = roc_curve(fold["y_test"], y_pred)
                auc = roc_auc_score(fold["y_test"], y_pred)

                evaluations["Accuracy"].append(acc)
                evaluations["Precision"].append(prec)
                evaluations["TPR"].append(tpr)
                evaluations["FPR"].append(fpr)
                evaluations["AUC"].append(auc)
                evaluations["Sensitivity"].append(sens)
                evaluations["Specificity"].append(spec)
                evaluations["MCC"].append(mcc)

            i = i+1

        with open("Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/"+m+"_evaluations_"+str(n_fold)+"fold_"+s+".pickle", "wb") as fp:
            pickle.dump(evaluations, fp)
        del model
        tf.keras.backend.clear_session()

        #Number of next species
        fNum=fNum+1


