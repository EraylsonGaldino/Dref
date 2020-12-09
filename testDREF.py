
import numpy as np


import DReFKDeepNONE
import DReFKDeepSARIMA2






#DREF ARIMA 
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('pollutn.txt'),12,12,'pollution_res_prev_conv_2.xlsx','pollution_res_prev_lstm.xlsx','Pollution')#20
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('gas.txt'),12,14,'gas_res_prev_conv_2.xlsx','gas_res_prev_lstm.xlsx','Gas')
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('lakeerie.txt'),12,100,'lake_res_prev_conv_2.xlsx','lake_res_prev_lstm.xlsx','Lake_Erie')
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('eletric.txt'),12,65,'elec_res_prev_conv_2.xlsx','elec_res_prev_lstm.xlsx','Electricity')
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('pigs.txt'),12,35,'pig_res_prev_conv_2.xlsx','pig_res_prev_lstm.xlsx','Pigs')
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('redwine.txt'),12,22,'redwine_res_prev_conv_2.xlsx','redwine_res_prev_lstm.xlsx','Red_wine_Corrected')
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('sunspotall.txt'),11,28,'sunspot_res_prev_conv_2.xlsx','sunspot_res_prev_lstm.xlsx','Sunspot')
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('carsales.txt'),12,20,'carsales_res_prev_conv_2.xlsx','carsales_res_prev_lstm.xlsx','Car_sales')
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('B1H.txt'),24,300,'b1h_res_prev_conv.xlsx','b1h_res_prev_lstm.xlsx','B1H')
#dd=DReFKDeepNONE.Dynamic(np.loadtxt('nordic.txt'),24,600,'nordic_res_prev_conv.xlsx','nordic_res_prev_lstm.xlsx','Nordic')







#DREF SARIMA
dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('pollutn.txt'),12,12,'poll_res_prev_conv_SARIMA.xlsx','poll_res_prev_lstm_SARIMA.xlsx','SARIMADREF_Pollution')#20
#dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('gas.txt'),12,14,'gas_res_prev_conv_SARIMA.xlsx','gas_res_prev_lstm_SARIMA.xlsx','SARIMADREF_Gas')
#dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('lakeerie.txt'),12,100,'lake_res_prev_conv_2_Sarima.xlsx','lake_res_prev_lstm_Sarima.xlsx','SARIMADREF_Lake_Erie')
#dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('eletric.txt'),12,65,'elec_res_prev_conv_SARIMA.xlsx','elec_res_prev_lstm_SARIMA.xlsx','SARIMADREF_Electricity')
#dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('pigs.txt'),12,35,'pigs_res_prev_conv_SARIMA.xlsx','pigs_res_prev_lstm_SARIMA.xlsx','SARIMADREF_Pigs')
#dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('redwine.txt'),12,22,'red_res_prev_conv_SARIMA.xlsx','red_res_prev_lstm_SARIMA.xlsx','SARIMADREF_Red_wine_Corrected')
#dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('sunspotall.txt'),11,28,'suns_res_prev_conv_SARIMA.xlsx','suns_res_prev_lstm_SARIMA.xlsx','SARIMADREF_Sunspot')
#dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('carsales.txt'),12,20,'car_res_prev_conv_SARIMA.xlsx','car_res_prev_lstm_SARIMA.xlsx','SARIMADREF_Car_sales')
#dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('B1H.txt'),24,300,'b1h_res_prev_conv_SARIMA.xlsx','b1h_res_prev_lstm_SARIMA.xlsx','SARIMADREF_B1H')
#dd=DReFKDeepSARIMA2.Dynamic(np.loadtxt('nordic.txt'),24,600,'nordic_res_prev_conv_SARIMA.xlsx','nordic_res_prev_lstm_SARIMA.xlsx','SARIMADREF_Nordic')


a = dd.start()




    

