"""
Plot Result

"""

import os
import pickle
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import interp
import tensorflow as tf
import evaluator
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score

species=["Elegans", "Melanogaster", "Sapiens", "Yeast"]
exp_dir = "Experiment5"
n_fold = 20
exp_NN = modName=["ConvLSTMCombV2"]#"Conv2Conv", "ConvLSTM", "Conv2ConvLSTMComb", "ConvStrideComb", "Comb3Layer", "ConvLSTMCombV2"]

i = 1
for nn in  exp_NN:

    fig = plt.figure(i,figsize=(12, 10))
    ax  = fig.add_subplot(111)

    accMean = []
    accStd = []
    mccMean = []
    mccStd = []
    sensMean = []
    sensStd = []
    specMean = []
    specStd = []

    for s in species:
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
        modelPath = "./Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/models/"+nn
        if(not os.path.exists("./Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/"+str(n_fold)+"fold_"+s+".pickle")):
               print("Error: Folds not Found")
        else:
            folds = []
            with open("./Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/"+str(n_fold)+"fold_"+s+".pickle", "rb") as fp:
                folds = pickle.load(fp)
            i = 1
            for fold in folds:
                #Check if model alredy exists
                if(os.path.exists(modelPath+"/"+nn+"_bestModel-fold"+str(i)+".hdf5")):  
                    input_size = (len(fold["X_test"][1]), 4)
                    # load json and create model
                    json_file = open(modelPath+"/"+nn+".json", 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    model = tf.keras.models.model_from_json(loaded_model_json)
                    model.build(input_shape=input_size)
                    model.summary()
                    # load weights into new model
                    model.load_weights(modelPath+"/"+nn+"_bestModel-fold"+str(i)+".hdf5")
                else:
                    print("Error: Model not Found")
                    break

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

        #with open("Result/"+s+"/"+exp_dir+"/"+str(n_fold)+"fold/"+nn+"_evaluations_"+str(n_fold)+"fold_"+s+".pickle", "rb") as fp:
        #    evaluations = pickle.load(fp)
        
        accMean.append(np.mean(evaluations["Accuracy"]))
        accStd.append(np.std(evaluations["Accuracy"]))
        mccMean.append(np.mean(evaluations["MCC"]))
        mccStd.append(np.std(evaluations["MCC"]))
        sensMean.append(np.mean(evaluations["Sensitivity"]))
        sensStd.append(np.std(evaluations["Sensitivity"]))
        specMean.append(np.mean(evaluations["Specificity"]))
        specStd.append(np.std(evaluations["Specificity"]))

    print(accStd)
    print(mccStd)
    print(sensStd)
    print(specStd)

    ind = np.arange(0, 4*len(species), step=4)    # the x locations for the groups
    width = 0.8        # the width of the bars
    p1 = ax.bar(ind, accMean, width, bottom=0, yerr=accStd)    

    p2 = ax.bar(ind+width, mccMean, width, bottom=0, yerr=mccStd)    

    p3 = ax.bar(ind+2*width, sensMean, width, bottom=0, yerr=sensStd)

    p4 = ax.bar(ind+3*width, specMean, width, bottom=0, yerr=specStd) 
    
    valueList = np.concatenate((np.around(accMean,3),np.around(mccMean,3),np.around(sensMean,3),np.around(specMean,3)))
    posList = np.concatenate((ind, (ind+width), (ind+2*width), (ind+3*width)))
    
   
    
    ax.set_title(nn+" Metrics")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(species)
    box = ax.get_position()
    ax.set_ylim(0,1.1)
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend((p1[0], p2[0], p3[0], p4[0]), ('Accuracy', 'MCC', 'Sensitivity', 'Specificity'), loc='center left', bbox_to_anchor=(1, 0.5))
    for i in range(len(valueList)):
        plt.text(x = posList[i]-0.5 , y = valueList[i]+0.05, s = valueList[i], size = 6, fontsize=9)
    ax.autoscale_view()

    plot_dir = "./Result/plot/"+nn+"/"+exp_dir
    if(not os.path.isdir(plot_dir)):
        os.makedirs(plot_dir)
    fig.savefig(plot_dir+"/"+nn+"_evaluations.png", bbox_inches='tight')
    fig.savefig(plot_dir+"/"+nn+"_evaluations.svg", format="svg", bbox_inches='tight')
    fig.savefig(plot_dir+"/"+nn+"_evaluations.eps", format="eps", bbox_inches='tight')

    i=i+1
     
