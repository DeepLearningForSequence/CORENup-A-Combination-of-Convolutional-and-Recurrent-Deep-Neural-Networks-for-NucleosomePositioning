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
from tensorflow.keras.models import load_model

def createModel(modCode, modName, fold, model, metricList = None):
    #Check if model alredy exists

    if(modCode == 3):
        modelLoaded = model.Conv_2Conv(input_size=input_size, metrics=metricsList)
    elif(modCode == 2): 
        modelLoaded = model.Conv_LSTMNN(input_size=input_size, metrics=metricsList)
    elif(modCode == 1):
        modelLoaded = model.ConvLSTMCombV2(input_size=input_size, metrics=metricsList)
        
    return modelLoaded

parser = argparse.ArgumentParser(description='Nucleosome Classification Experiment')
parser.add_argument('-m', '--models', dest = 'modCodes', choices=[1,2,3], type=int, nargs='+', default=[3],
                    help='NN Models name')
parser.add_argument('-p','--path', dest='path', type=str, default="/home/amato/Scrivania/CORENup/Datasets/Setting2/Yeast/pickle",
                    help='Pickle file Path')
parser.add_argument('-n','--nuc', dest='nucPickle', type=str, default="encoded_nuc_Yeast_wg.pickle",
                    help='Nucleosome filename')
parser.add_argument('-l','--lin', dest='linkPickle', type=str, default="encoded_link_Yeast_wg.pickle",
                    help='Linker filename')
parser.add_argument('-o','--out', dest='outPath', type=str, default="/home/amato/Scrivania/CORENup/Datasets/Setting2/Yeast/",
                    help='Output file Path')
parser.add_argument('-e', '--experiments', dest='exp' , default='Experiment1',
                    help='Experiments Name')
parser.add_argument('-nf', '-nfolds', dest='nFolds', default=1,
                    help='Number of Folds')
parser.add_argument('-f', '-foldName', dest='foldName', default="folds.pickle",
                    help='Folds Filename')

args = parser.parse_args()
n_fold = args.nFolds
inPath = args.path
nucPickle = args.nucPickle
linkPickle = args.linkPickle
outPath = args.outPath
expName = args.exp
models = args.modCodes
foldName = args.foldName

metricsList = [evaluator.precision, evaluator.recall, evaluator.f1score]

epochs=200
batch_size = 64
shuffle = False #True
seed = None #1001


#Number of the species for Roc Curve Figure
fNum = 1

if(os.path.exists(os.path.join(outPath,"elapsed.json"))):
    os.path.join(outPath,"elapsed.json")

modName=["ConvLSTMCombV2", "Conv_LSTMNN", "Conv2Conv"]


for mCode in models:
    m = modName[mCode-1]
    print(m)
    #Create and set model save dir
    modelPath = os.path.join(outPath, expName, "{}fold".format(n_fold), "models", m)
    if(not os.path.isdir(modelPath)):
        os.makedirs(modelPath)
    

    #Load nucleosome and linker and then create dataset and  class labels
    nucList = []
    linkList = []
    with open(os.path.join(inPath, nucPickle), "rb") as fp:
        nucList = pickle.load(fp)
    with open(os.path.join(inPath, linkPickle), "rb") as fp:
        linkList = pickle.load(fp)

    input_size = (len(nucList[1]), 4)
    labels = np.concatenate((np.ones((len(nucList), 1), dtype=np.float32), np.zeros((len(linkList), 1), dtype=np.float32)), axis=0)
    dataset = np.concatenate((nucList,linkList),0)

    #Create folder to save fold dataset and build kfold
    foldPath = os.path.join(outPath, expName, "{}fold".format(n_fold), foldName)
    if(not os.path.exists(foldPath)):
        folds = evaluator.build_kfold(dataset, labels, k=20, shuffle=shuffle, seed=seed)
        with open(foldPath, "wb") as fp:
            pickle.dump(folds,fp)
    else:
        with open(foldPath, "rb") as fp:
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
        if(not os.path.exists(os.path.join(modelPath, "{}_bestModel-fold{}.hdf5".format(m, i)))):
            #If not exist then create the model and fit it with the fold
            modelCallbacks = [
                tf.keras.callbacks.ModelCheckpoint(os.path.join(modelPath, "{}_bestModel-fold{}.hdf5".format(m, i)), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
            ]
            modelClass = modelsNN.modelsClass()
            model = createModel(mCode, m, i, modelClass, metricsList)
            print(model.summary())
            model.fit(x=fold["X_train"], y=fold["y_train"], batch_size=batch_size, epochs=epochs, verbose=1, callbacks=modelCallbacks, validation_data=(fold["X_test"], fold["y_test"]), validation_freq=1)
        
        else:
            
            #If exists then load the model and performe evaluation
            print("Modello esistente per fold: "+str(i))
            model = load_model(os.path.join(modelPath, "{}_bestModel-fold{}.hdf5".format(m, i)), 
                                custom_objects={"precision":evaluator.precision, "recall":evaluator.recall, "f1score":evaluator.f1score})
            print(model.summary())
            # load weights into new model
            model.load_weights(modelPath+"/"+m+"_bestModel-fold"+str(i)+".hdf5")

            """
            
            2)Evaluate Model on Fold i

            """

            print("Prevedo le Classi per Fold{}".format(i))
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

    with open(os.path.join(foldPath, "{}_evaluations_{}fold.pickle".format(m,n_fold)), "wb") as fp:
        pickle.dump(evaluations, fp)
    del model
    tf.keras.backend.clear_session()


