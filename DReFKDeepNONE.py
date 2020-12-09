import pandas as pd
import numpy as np
import openpyxl
from statsmodels.tsa.stattools import acf
import rpy2.robjects as r
import rpy2.robjects.numpy2ri
from sklearn.metrics import mean_squared_error
rpy2.robjects.numpy2ri.activate()
import DataHandlerDReF as dh
from sklearn.neural_network import MLPRegressor
import sklearn.neighbors as nn
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import jensenshannon
from sklearn.svm import SVR
from RBF import RBFNet
from fastdtw import fastdtw
from scipy.spatial import distance
from scipy.stats import entropy
from shapedtw import matching_shapedtw
from collections import OrderedDict
import time

class Dynamic:
    def __init__(self,data,window,testNO,fileCONV,fileLSTM,dataname):
        self.data=data
        self.dimension=window
        self.ndata=(data-min(data))/(max(data)-min(data))
        self.trainset=0.6
        self.valset=0.4
        self.testNO=testNO
        self.fileCONV=fileCONV
        self.fileLSTM=fileLSTM
        self.dataname=dataname
    def readXLXs(self,fileDBN):

        wb = openpyxl.load_workbook(filename=fileDBN)
        sheet_ranges = wb.get_sheet_by_name('Treinamento')
        treino=[]
        cont=0
        for i in sheet_ranges.rows:
            if cont!=0:
                treino.append(i[2].value)
            cont=cont+1

        sheet_ranges = wb.get_sheet_by_name('Validação 1')
        val1=[]
        cont=0
        for i in sheet_ranges.rows:
            if cont!=0:
                val1.append(i[2].value)
            cont=cont+1

        sheet_ranges = wb.get_sheet_by_name('Validação 2')
        val2=[]
        cont=0
        for i in sheet_ranges.rows:
            if cont!=0:
                val2.append(i[2].value)
            cont=cont+1

        sheet_ranges = wb.get_sheet_by_name('Teste')
        test=[]
        cont=0
        for i in sheet_ranges.rows:
            if cont!=0:
                test.append(i[2].value)
            cont=cont+1

        return (np.asarray(treino),np.asarray(val1),np.asarray(val2),np.asarray(test))

    def start(self):
        time_train_arima=0
        time_test_arima=0
        time_train_MLP=0
        time_test_MLP=0
        time_train_SVR=0
        time_test_SVR=0
        time_train_RBF = 0
        time_test_RBF=0
        time_test_DREF=0
        dh2 = dh.DataHandler(self.ndata, self.dimension, self.trainset, self.valset, self.testNO)
        train_set, train_target, val_set, val_target, val_setf, val_targetf, test_set, test_target, arima_train, arima_val,arima_valf, arima_test = dh2.redimensiondata(
            self.ndata, self.dimension, self.trainset, self.valset, self.testNO)
        valNO=len(arima_valf)
        r.r('library(forecast)')
        ets = r.r('auto.arima')
        numeric = r.r('as.numeric')
        inicio = time.time()
        fit = ets(np.array(arima_train))
        fim = time.time()
        time_train_arima=fim-inicio
        fitted = r.r('fitted')
        arimaorder = r.r('arimaorder')
        ordemModelo = arimaorder(fit)
        print('ARIMA(%d, %d, %d)'%(ordemModelo[0],ordemModelo[1],ordemModelo[2]))
        predTreino = fitted(fit)
        residTreino=np.asarray(arima_train)-np.asarray(predTreino)
        ArimaTest = r.r('Arima')
        fitval= ArimaTest(np.array(arima_val),  model=fit)
        predVal=np.asarray(fitted(fitval))
        residVal=np.asarray(arima_val)-np.asarray(predVal)
        fitvalf2= ArimaTest(np.array(arima_valf),  model=fit)
        predValf2=np.asarray(fitted(fitvalf2)) 
        residVal2=np.asarray(arima_valf)-np.asarray(predValf2)
        inicio = time.time()
        fittest = ArimaTest(np.array(arima_test), model=fit)
        fim = time.time()
        time_test_arima=fim-inicio
        predTest =np.asarray( fitted(fittest))
        residTest=np.asarray(arima_test)-np.asarray(predTest)
        
        print('resíduos Treino \n')
        
        print(residTreino.tolist())
        print('\n resíduos val \n')
        print(residVal.tolist())
        print('\n resíduos val2 \n')
        print(residVal2.tolist())
        print('\n resíduos teste \n')
        print(residTest.tolist())
        
        print('\n \n')
        predAll=[]
        predAll.extend(np.asarray(predTreino))
        predAll.extend(np.asarray(predVal))
        predAll.extend(np.asarray(predValf2))
        predAll.extend(np.asarray(predTest))
        nlgs=self.dimension*2
        acs,confs= acf(np.asarray(arima_train)-np.asarray(predTreino),nlags=nlgs,alpha=0.05)
        newlags=[]
        for i in range(nlgs+1):
            if (confs[i][0]>acs[i] or acs[i]>confs[i][1]):
                newlags.append(i)

        resids2 = np.asarray(self.ndata) - np.asarray(predAll)
        resids = (resids2-min(resids2))/(max(resids2)-min(resids2))
        dh3 = dh.DataHandler(resids, self.dimension, self.trainset, self.valset, self.testNO)

        train_set2, train_target2, val_set2, val_target2,val_setf2, val_targetf2, test_set2, test_target2, arima_train2, arima_val2,arima_valf2, arima_test2 = dh3.redimensiondata(
            resids, self.dimension, self.trainset, self.valset, self.testNO)

        nn1 = MLPRegressor(activation='logistic', solver='lbfgs', shuffle=False)
        
        rna = GridSearchCV(nn1, param_grid={
            'hidden_layer_sizes': [(2,), (5,), (10,), (15,), (20,)]})
        
        
        inicio = time.time()
        rna.fit(train_set2, train_target2)
        fim=time.time()
        time_train_MLP = fim-inicio

        predRNATreino = rna.predict(train_set2)
        predRNAVal = rna.predict(val_set2)
        predRNAVal2=rna.predict(val_setf2)
        inicio=time.time()
        predRNATest= rna.predict(test_set2)
        fim=time.time()
        time_test_MLP = fim-inicio
        predALLRNA=[]
        predALLRNA.extend(np.asarray(predRNATreino))
        predALLRNA.extend(np.asarray(predRNAVal))
        predALLRNA.extend(np.asarray(predRNATest))

        svr1=SVR(kernel='rbf', max_iter=-1, shrinking=True, verbose=False)
        inicio = time.time()
        svr = GridSearchCV(svr1, param_grid={
            'C': [0.1,1,100,1000,10000],
             'gamma':[1,0.1,0.01,0.001],
            'epsilon':[0.1, 0.01, 0.001]})

        #regcomb = linear_model.LinearRegression()
        inicio = time.time()
        svr.fit(train_set2, train_target2)
        fim = time.time()
        time_train_SVR = fim - inicio
        predSVRTreino = svr.predict(train_set2)
        predSVRVal = svr.predict(val_set2)
        predSVRVal2 = svr.predict(val_setf2)
        inicio=time.time()
        predSVRTeste= svr.predict(test_set2)
        fim = time.time()
        time_test_SVR = fim-inicio

        #n_hidden
        rbf=RBFNet(k=3)
       # rbf = GridSearchCV(rbf1, param_grid={
        #    'k': [3, 5, 10, 15]})
        inicio=time.time()
        rbf.fit(np.asarray(train_set2), np.asarray(train_target2))
        fim=time.time()
        time_train_RBF = fim-inicio

        #elmsts=GridSearchCV(elms, param_grid={
        #    'activation_func':['sigmoid','tanh','softlim'],
        #    'n_hidden': [5,10,15,20,25,30,40,50]})
        #elms.fit(np.asarray(train_set2),np.asarray(train_target2))
        predArimaTreino=rbf.predict(np.asarray(train_set2))[:,0]
        predArimaVal=rbf.predict(np.asarray(val_set2))[:,0]
        predArimaValf2=rbf.predict(np.asarray(val_setf2))[:,0]
        inicio = time.time()
        predArimaTest=rbf.predict(np.asarray(test_set2))[:,0]
        fim = time.time()
        time_test_RBF = fim-inicio
#        arima2 = ets(numeric(arima_train2))
#        predArimaTreino = fitted(arima2)
#        ArimaVal = (ArimaTest(numeric(arima_val2),  model=arima2))
#        predArimaVal = np.asarray(fitted(ArimaVal))
#        ArimaVal2 = (ArimaTest(numeric(arima_valf2),  model=arima2))
#        predArimaValf2 = np.asarray(fitted(ArimaVal2))
#        ArimaTest = (ArimaTest(numeric(arima_test2),  model=arima2))
#        predArimaTest = np.asarray(fitted(ArimaTest))
        
        
        
        predRNAVal2 = predRNAVal2*(max(resids2)-min(resids2)) +min(resids2)
        predSVRVal2 = predSVRVal2*(max(resids2)-min(resids2)) +min(resids2)
        predArimaValf2 = predArimaValf2*(max(resids2)-min(resids2)) +min(resids2)
        val_targetf2=np.asarray(val_targetf2)
        val_targetf2=val_targetf2*(max(resids2)-min(resids2)) +min(resids2)
        predValf2=predValf2#*(max(resids2)-min(resids2)) +min(resids2)
        
        predRNATest= predRNATest*(max(resids2)-min(resids2)) +min(resids2)
        predSVRTeste= predSVRTeste*(max(resids2)-min(resids2)) +min(resids2)
        predArimaTest=predArimaTest*(max(resids2)-min(resids2)) +min(resids2)
        test_target2=np.asarray(test_target2)
        test_target2=test_target2*(max(resids2)-min(resids2)) +min(resids2)

 #       (predDBNtreino, predDBNval1, predDBNval2, predDBNtest)= self.readXLXs(self.fileDBN)
        (predLSTMtreino, predLSTMval1, predLSTMval2, predLSTMtest) = self.readXLXs(self.fileLSTM)
        (predCONVtreino, predCONVval1, predCONVval2, predCONVtest) = self.readXLXs(self.fileCONV)

        predLSTMval1= (predLSTMval1-min(resids2))/(max(resids2)-min(resids2))
        predCONVval1= (predCONVval1-min(resids2))/(max(resids2)-min(resids2))

        val_target2 = np.asarray(val_target2)
        val_target = np.asarray(val_target)
        K=0
        def simule_metrics(metrica):

            Ks=[1,3,5,10,15] # Adicionar o 3
            mseDyn=[]
            taxaacertoval=[]
            erroval=[]
        
            acertoval=0
            for kv in Ks: #seleção do K usando duas validações
                acertoval=0
                neigh = nn.NearestNeighbors(kv, metric=metrica,algorithm='auto')
                if kv==3:
                    print('parou')

                neigh.fit(val_set2)

                (distr, knearest) = neigh.kneighbors(val_setf2)
                test_prediction = []
                oraclepred=[]
                oraclepredlow=[]
                
                for conts in range(valNO):
                    predSVRResVal = ((predSVRVal[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
                    predRnaResVal = ((predRNAVal[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
                    predARIMAResVal =((predArimaVal[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
                    predCONVResVal = ((predCONVval1[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
                    predLSTMResVal = ((predLSTMval1[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
                    predsNOMETHOD = (((predVal[knearest]))[conts])
     #               predDBNResVal = ((predDBNval1[knearest])[conts])

                    mseSVRRES = mean_squared_error((val_target[knearest])[conts], predSVRResVal+predsNOMETHOD)
                    mseRnaRES = mean_squared_error((val_target[knearest])[conts], predRnaResVal+predsNOMETHOD)
                    mseArimaRES = mean_squared_error((val_target[knearest])[conts], predARIMAResVal+predsNOMETHOD)
                    mseCONVRes = mean_squared_error((val_target[knearest])[conts], predCONVResVal+predsNOMETHOD)
                    mseLSTMRes = mean_squared_error((val_target[knearest])[conts], predLSTMResVal+predsNOMETHOD)
                    mseNOMETHODres = mean_squared_error((val_target[knearest])[conts],predsNOMETHOD)
    #                mseDBNRes = mean_squared_error((val_target2[knearest])[conts], predDBNResVal)

                    mses = [mseSVRRES, mseRnaRES, mseArimaRES,mseCONVRes,mseLSTMRes,mseNOMETHODres]
                    index = np.argmin(mses)


                    if index == 2:

                        test_prediction.append((predValf2[conts]) + ((predArimaValf2[conts])))
                    if index ==1:
                        test_prediction.append((predValf2[conts]) + ((predRNAVal2[conts])))
                    if index==0:
                        test_prediction.append((predValf2[conts]) + ((predSVRVal2[conts])))

                    if index == 3:
                        test_prediction.append((predValf2[conts]) + ((predCONVval2[conts])))

                    if index == 4:
                        test_prediction.append((predValf2[conts]) + ((predLSTMval2[conts])))
                    if index == 5:
                        test_prediction.append((predValf2[conts]))

                    erroval= (np.abs(val_target2[conts]- (predSVRVal2[conts]+predValf2[conts])),np.abs(val_target2[conts]- (predRNAVal2[conts]+predValf2[conts])),np.abs(val_target2[conts]- (predArimaValf2[conts]+predValf2[conts])),np.abs(val_target2[conts]- (predCONVval2[conts]+predValf2[conts])),np.abs(val_target2[conts]- (predLSTMval2[conts]+predValf2[conts])),np.abs(val_target2[conts]-predValf2[conts]))

                    indexoracle=np.argmin(erroval)
                    if(index==indexoracle):
                        acertoval=acertoval+1

    #                if indexoracle == 0:
    #                    oraclepred.append(predRNATest[cont]+predTest[cont])
    #                if indexoracle == 1:
    #                    oraclepred.append(predSVRTeste[cont]+predTest[cont])
    #                if indexoracle == 2:
    #                    oraclepred.append(predArimaTest[cont]+predTest[cont])
    #
    #                indexoraclelow=np.argmax(erro)
    #                if indexoraclelow == 0:
    #                    oraclepredlow.append(predRNATest[cont]+predTest[cont])
    #                if indexoraclelow == 1:
    #                    oraclepredlow.append(predSVRTeste[cont]+predTest[cont])
    #                if indexoraclelow == 2:
    #                    oraclepredlow.append(predArimaTest[cont]+predTest[cont])

                taxaacertoval.append(acertoval/valNO)
                mseDyn.append(mean_squared_error(test_prediction,val_targetf))
                #print('VALIDATION  Taxa acerto val K=%d :   %f '%(kv,acertoval/valNO))
                #print(' _________________________________________\n\n')
            K=Ks[np.argmin(mseDyn)]
            neigh = nn.NearestNeighbors(K, metric=metrica,algorithm='auto')
            SvrCont=0
            RNACont=0
            ARIMACont=0
            NOMETHODCont=0
            CONVcont=0
            LSTMcont=0
            DBNcont=0
            lowSvrCont=0
            lowRNACont=0
            lowARIMACont=0
            lowCONVCont = 0
            lowLSTMCont = 0
            lowNOMETHOD=0
            lowDBNCont = 0
            acerto=0.0
            lcont=[]
            neigh.fit(val_set2)

            RNACONT2 =0
            SVRCONT2=0
            ARIMACONT2=0
            LSTMCONT2=0
            CONVCONT2=0
            DBNCONT2=0
            NOMETHODCONT2=0

            (distr, knearest) = neigh.kneighbors(test_set2)
            test_prediction = []
            oraclepred=[]
            oraclepredlow=[]
            inicio = time.time()
            for conts in range(self.testNO):
                predSVRResVal = ((predSVRVal[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
                predRnaResVal = ((predRNAVal[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
                predARIMAResVal =((predArimaVal[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
                predCONVResVal = ((predCONVval1[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
    #            predDBNResVal = ((predDBNval1[knearest])[conts])
                predLSTMResVal = ((predLSTMval1[knearest])[conts])*(max(resids2)-min(resids2)) +min(resids2)
                predsNOMETHOD = ((predVal[knearest])[conts])
                mseSVRRES = mean_squared_error((val_target[knearest])[conts], predSVRResVal+predsNOMETHOD)
                mseRnaRES = mean_squared_error((val_target[knearest])[conts], predRnaResVal+predsNOMETHOD)
                mseArimaRES = mean_squared_error((val_target[knearest])[conts], predARIMAResVal+predsNOMETHOD)
                mseCONVRES = mean_squared_error((val_target[knearest])[conts], predCONVResVal+predsNOMETHOD)
    #            mseDBNRES = mean_squared_error((val_target2[knearest])[conts], predDBNResVal)
                mseLSTMRES = mean_squared_error((val_target[knearest])[conts], predLSTMResVal+predsNOMETHOD)
                mseNOMETHOD = mean_squared_error((val_target[knearest])[conts],predsNOMETHOD)

                mses = [mseSVRRES, mseRnaRES, mseArimaRES,mseCONVRES,mseLSTMRES,mseNOMETHOD]
                index = np.argmin(mses)


                if index == 2:

                    test_prediction.append((predTest[conts]) + ((predArimaTest[conts])))
                    ARIMACONT2=ARIMACONT2+1
                if index ==1:
                    test_prediction.append((predTest[conts]) + ((predRNATest[conts])))
                    RNACONT2=RNACONT2+1
                if index==0:
                    test_prediction.append((predTest[conts]) + ((predSVRTeste[conts])))
                    SVRCONT2=SVRCONT2+1
                if index == 3:
                    test_prediction.append((predTest[conts]) + ((predCONVtest[conts])))
                    CONVCONT2 = CONVCONT2 + 1
                if index == 4:
                    test_prediction.append((predTest[conts]) + ((predLSTMtest[conts])))
                    LSTMCONT2 = LSTMCONT2 + 1
                if index ==5:
                    test_prediction.append((predTest[conts]) )
                    NOMETHODCONT2=NOMETHODCONT2+1


                erro= (np.abs(test_target[conts]- (predSVRTeste[conts]+predTest[conts])),np.abs(test_target[conts]- (predRNATest[conts]+predTest[conts])),np.abs(test_target[conts]- (predArimaTest[conts]+predTest[conts])),np.abs(test_target[conts]- (predCONVtest[conts]+predTest[conts])),np.abs(test_target[conts]- (predLSTMtest[conts]+predTest[conts])),np.abs(test_target[conts]- predTest[conts]))
                #print(erro)
                indexoracle=np.argmin(erro)

                if indexoracle == 1:
                    oraclepred.append(predRNATest[conts]+predTest[conts])
                    RNACont=RNACont+1
                if indexoracle == 0:
                    oraclepred.append(predSVRTeste[conts]+predTest[conts])
                    SvrCont=SvrCont+1
                if indexoracle == 2:
                    oraclepred.append(predArimaTest[conts]+predTest[conts])
                    ARIMACont=ARIMACont+1
                if indexoracle == 3:
                    oraclepred.append(predCONVtest[conts] + predTest[conts])
                    CONVcont = CONVcont + 1
                if indexoracle == 4:
                    oraclepred.append(predLSTMtest[conts] + predTest[conts])
                    LSTMcont = LSTMcont + 1
                if indexoracle == 5:
                    oraclepred.append( predTest[conts])
                    NOMETHODCont = NOMETHODCont + 1


                lcont.append((index,indexoracle))
                indexoraclelow=np.argmax(erro)
                if(indexoracle==index):
                    acerto=acerto+1.0
                if indexoraclelow == 1:
                    oraclepredlow.append(predRNATest[conts]+predTest[conts])
                    lowRNACont=lowRNACont+1
                if indexoraclelow == 0:
                    oraclepredlow.append(predSVRTeste[conts]+predTest[conts])
                    lowSvrCont=lowSvrCont+1
                if indexoraclelow == 2:
                    oraclepredlow.append(predArimaTest[conts]+predTest[conts])
                    lowARIMACont=lowARIMACont+1
                if indexoraclelow == 3:
                    oraclepredlow.append(predCONVtest[conts]+predTest[conts])
                    lowCONVCont=lowCONVCont+1
                if indexoraclelow == 4:
                    oraclepredlow.append(predLSTMtest[conts]+predTest[conts])
                    lowLSTMCont=lowLSTMCont+1
                if indexoraclelow == 5:
                    oraclepredlow.append(predTest[conts])
                    lowNOMETHOD=lowNOMETHOD+1
            fim=time.time()
          
            mseDReF = mean_squared_error(test_prediction,test_target)
            mseARIMAf = mean_squared_error(predTest,test_target)
            mseRNA = mean_squared_error((predRNATest+predTest),test_target)
            mseSVR = mean_squared_error((predSVRTeste+predTest),test_target)

            mseCONV = mean_squared_error((predCONVtest + predTest), test_target)
            #mseDBN = mean_squared_error((predDBNtest + predTest), test_target)
            mseLSTM = mean_squared_error((predLSTMtest + predTest), test_target)

            mseARIMA = mean_squared_error((predArimaTest+predTest),test_target)
            mseOracle = mean_squared_error(oraclepred,test_target)
            mseLower =  mean_squared_error((oraclepredlow),test_target)
            print('\n\n ---- Test MSE ---- \n\n')
            print('ARIMA               = %f'%(mseARIMAf))
            print('ARIMA + RNA         = %f'%(mseRNA))
            print('ARIMA + SVR         = %f'%(mseSVR))
            print('ARIMA + RBF         = %f'%(mseARIMA))
            print('ARIMA + LSTM         = %f'%(mseLSTM))
            #print('ARIMA + DBN         = %f'%(mseDBN))
            print('ARIMA + CONV         = %f'%(mseCONV))
            print('ARIMA + DReF (K=%d)  = %f'%(K,mseDReF))
            print('Upper Bound         = %f'%(mseLower))
            print('Lower Bound         = %f'%(mseOracle))
            print('Taxa de acerto      = %f'%(acerto/self.testNO))
            print('Taxa RNA            = %f'%(RNACONT2/self.testNO))
            print('Taxa SVR            = %f' % (SVRCONT2 / self.testNO))
            print('Taxa RBF          = %f' % (ARIMACONT2 / self.testNO))
            print('Taxa CONV            = %f'%(CONVCONT2/self.testNO))
           # print('Taxa DBN            = %f' % (DBNCONT2 / self.testNO))
            print('Taxa LSTM            = %f' % (LSTMCONT2 / self.testNO))
            print('Taxa NOMETHOD        =%f'%(NOMETHODCONT2/self.testNO))
            df2Dict = OrderedDict({'Methods':['ARIMA','ARIMA + RNA','ARIMA + SVR','ARIMA + RBF','ARIMA + LSTM','ARIMA + CONV','ARIMA + DReF (K=%d)'%(K),'Upper Bound','Lower Bound','Hit Rate','Taxa RNA','Taxa SVR','Taxa RBF','Taxa CONV ','Taxa LSTM','Taxa NOMETHOD']
            ,'Values':[mseARIMAf,mseRNA,mseSVR,mseARIMA,mseLSTM,mseCONV,mseDReF,mseLower,mseOracle,acerto/self.testNO,RNACONT2/self.testNO,SVRCONT2 / self.testNO,ARIMACONT2 / self.testNO,CONVCONT2/self.testNO,LSTMCONT2 / self.testNO,NOMETHODCONT2/self.testNO]})
            df2 = pd.DataFrame(df2Dict)
            upperDict=OrderedDict({'Upper  selection': ['RBF',    'SVR' ,     'RNA'      ,'CONV'       ,'LSTM ', 'NOMETHOD'],'Rates':[lowARIMACont/self.testNO,lowSvrCont/self.testNO,lowRNACont/self.testNO,lowCONVCont/self.testNO,lowLSTMCont/self.testNO,lowNOMETHOD/self.testNO]})
            selectionUPPER=pd.DataFrame(upperDict)
            print('Upper  selection: RBF = %f    SVR = %f     RNA = %f      CONV = %f       LSTM = %f       NOMETHOD = %f'%(lowARIMACont/self.testNO,lowSvrCont/self.testNO,lowRNACont/self.testNO,lowCONVCont/self.testNO,lowLSTMCont/self.testNO,lowNOMETHOD/self.testNO))
            lowDict = OrderedDict({'Lower  selection': ['RBF',    'SVR' ,     'RNA'      ,'CONV'       ,'LSTM ', 'NOMETHOD'],'Rates':[ARIMACont/self.testNO,SvrCont/self.testNO,RNACont/self.testNO,CONVcont/self.testNO,LSTMcont/self.testNO,NOMETHODCont/self.testNO]})
            selectionLOWER=pd.DataFrame(lowDict)
          
            print('Lower  selection: RBF = %f    SVR = %f     RNA = %f      CONV = %f       LSTM = %f        NOMETHOD = %f'%(ARIMACont/self.testNO,SvrCont/self.testNO,RNACont/self.testNO,CONVcont/self.testNO,LSTMcont/self.testNO,NOMETHODCont/self.testNO))
            
            #print('\n\n ------ Seleções 0-SVR  1-RNA   2-RBF  3- CONV  4 - LSTM ------------ \n\n')
            #for i in range(np.size(test_target)):
             #   print('%d  \t  %d  \t'%(lcont[i][0],lcont[i][1]))
            #print('\n---------------------------------\n\n')

            print('\n\n\t\t Previsões \n\n')
            print('TARGET   \t  ARIMA     \t  ARIMA+RNA \t ARIMA+SVR \t  ARIMA+RBF \t  ARIMA+CONV \t ARIMA+LSTM \t  ARIMA+DReF \t   Oracle')
            for i in range(np.size(test_target)):
                print('%f  \t  %f  \t  %f \t     %f \t  %f  \t  %f  \t      %f  \t  %f  \t   %f'%(test_target[i],predTest[i],predRNATest[i]+predTest[i],predSVRTeste[i]+predTest[i],predArimaTest[i]+predTest[i],predCONVtest[i]+predTest[i],predLSTMtest[i]+predTest[i],test_prediction[i],oraclepred[i]))
            r1=np.arange(len(mseDyn))
            predsDict = OrderedDict({'TARGET':test_target,   '  ARIMA ':predTest,    '  ARIMA+RNA ':predRNATest+predTest,'ARIMA+SVR ':predSVRTeste+predTest, 'ARIMA+RBF ':predArimaTest+predTest,   'ARIMA+CONV ':predCONVtest+predTest, 'ARIMA+LSTM ':predLSTMtest+predTest,  'ARIMA+DReF ':np.array(test_prediction),  ' Oracle':np.array(oraclepred)})
            trestPrediction2 = np.array(test_prediction)
            oraclePred2=np.array(oraclepred)
            acerto2=0
            for ds in range(self.testNO):
                if (trestPrediction2[ds]==oraclePred2[ds]):
                    acerto2=acerto2+1.0
            if acerto2!=acerto:
                print('######################################\n######################################################################\n######################################################################\n######################################################################\n################################\n pare \n')
                    
            dfPrevs=pd.DataFrame(predsDict)
            #r2=[x+0.25 for x in r1]
            #plt.bar(r1,mseDyn,width=0.25)
            #plt.ylabel('MSE',fontweight='bold')
            #plt.xlabel('Number of neighbors', fontweight='bold')
            #plt.xticks([r+0.25/2 for r in range(len(mseDyn))],['K=1','K=3','K=5','K=10','K=15'])
            #plt.legend(['Validation'])
            #plt.show()

            print('\n\n---   Validation   --- \n K=1 \t %f\n K=3 \t %f\n K=5 \t %f\n K=10 \t %f\n K=15 \t %f\n\n\n ----------------------'%(mseDyn[0],mseDyn[1],mseDyn[2],mseDyn[3],mseDyn[4]))
            valDict = OrderedDict({'K':['K=1','K=3','K=5','K=10','K=15'],'MSE':[mseDyn[0],mseDyn[1],mseDyn[2],mseDyn[3],mseDyn[4]]})
            dfValidation=pd.DataFrame(valDict)
            
            return df2,selectionUPPER,selectionLOWER,dfPrevs,dfValidation,K
        def dtw(a, b):

            d, p = fastdtw(a, b)
            return d

        def Chebyshev(a, b):

            return distance.chebyshev(b, a)

        def euclidian(a, b):

            return distance.euclidean(a, b)

        def shape_dtw(a, b):

            a = np.reshape(a, (a.shape[0], 1))
            b = np.reshape(b, (b.shape[0], 1))

            dist, correspondences = matching_shapedtw(a, b, euclidian)
            return dist

        def correlation(a, b):

            return distance.correlation(a, b)

        mseEU,upperEU,lowerEU,prevsEU,valEU,K= simule_metrics('euclidean')
        mseDTW,upperDTW,lowerDTW,prevsDTW,valDTW,K=simule_metrics(dtw)
        mseCHEB,upperCHEB,lowerCHEB,prevsCHEB,valCHEB,K=simule_metrics(Chebyshev)
        mseSHP,upperSHP,lowerSHP,prevsSHP,valSHP,K1=simule_metrics(shape_dtw)
        mseCOR,upperCOR,lowerCOR,prevsCOR,valCOR,K=simule_metrics(correlation)
        mseMUT,upperMUT,lowerMUT,prevsMUT,valMUT,K=simule_metrics(jensenshannon)
        mseENT,upperENT,lowerENT,prevsENT,valENT,K=simule_metrics(entropy)
        writer = pd.ExcelWriter('%s_TESTEV2.xlsx'%(self.dataname), engine='xlsxwriter')
        workbook = writer.book
        def relatorios(namesheet,df1,df2,df3,df4,df5):
            worksheet = workbook.add_worksheet(namesheet)
            writer.sheets[namesheet] = worksheet
            df1.to_excel(writer, sheet_name=namesheet, startrow=0, startcol=0)
            df2.to_excel(writer, sheet_name=namesheet, startrow=20, startcol=0)
            df3.to_excel(writer, sheet_name=namesheet, startrow=28, startcol=0)
            df4.to_excel(writer, sheet_name=namesheet, startrow=36, startcol=0)
            df5.to_excel(writer, sheet_name=namesheet, startrow=43, startcol=0)
        relatorios('Euclidean',mseEU,upperEU,lowerEU,valEU,prevsEU)
        relatorios('DTW',mseDTW,upperDTW,lowerDTW, valDTW, prevsDTW)
        relatorios('Chebychev', mseCHEB,upperCHEB,lowerCHEB, valCHEB, prevsCHEB)
        relatorios('Shape_DTW', mseSHP,upperSHP,lowerSHP, valSHP, prevsSHP)
        relatorios('Correlation', mseCOR,upperCOR,lowerCOR, valCOR, prevsCOR)
        relatorios('JensenShannon', mseMUT,upperMUT,lowerMUT, valMUT, prevsMUT)
        relatorios('Entropy', mseENT,upperENT,lowerENT, valENT, prevsENT)
        writer.close()


        return (min([mseSHP['Values'][6],mseEU['Values'][6],mseCHEB['Values'][6],mseMUT['Values'][6],mseENT['Values'][6]]),K1)
        #return (mseARIMAf,mseRNA,mseSVR,mseARIMA,mseDyn[0],mseDyn[1],mseDyn[2],mseDyn[3],mseDyn[4],mseLower,mseOracle)
        
    def createLags(self,nlags,target,pred1,pred2,testNo,trainset):
        sa=len(target)
        sb=len(pred1)
        sc=len(pred2)
        v=(sa,sb,sc)
        index =np.argmin(v)
        targetf=target[sa-v[index]:len(target)]
        pred1f = pred1[sb-v[index]:len(pred1)]
        pred2f = pred2[sc-v[index]:len(pred2)]
        s1=pd.Series(pred1f)
        s2=pd.Series(pred2f)
        s3=pd.Series(targetf)
        res1 = pd.concat([s1.shift(i) for i in range(0, nlags)], axis=1)
        res2 = pd.concat([s2.shift(i) for i in range(0, nlags)], axis=1)

        resf=pd.concat([res1,res2],axis=1)

        test = resf.iloc[v[index] - testNo:v[index], :]
        test_target = s3.iloc[v[index] - testNo:v[index]]



        idrem=v[index] - testNo
        tra=np.int32(np.floor(trainset*idrem))

        train = resf.iloc[nlags:tra,:]
        train_target = s3.iloc[nlags:tra]

        val = resf.iloc[tra:idrem,:]
        val_target = s3.iloc[tra:idrem]

        return (np.array(train),np.array(train_target),np.array(val),np.array(val_target),np.array(test),np.array(test_target))

