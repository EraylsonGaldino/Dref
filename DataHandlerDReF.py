import pandas as pd
import numpy


class DataHandler:
    def __init__(self, data, dimension, trainset, valset,testNO):
        self.data=data
        self.dimension=dimension
        self.trainset=trainset
        self.valset=valset -0.1
        self.valset2=0.1
        self.ndata=(data-min(data))/(max(data)-min(data))
        self.testNO=testNO
    def redimensiondata(self, data, dimension, trainset, valset,testNO):
        valset=valset-0.1
      
        s1 = pd.Series(data)
        lin2 = len(data)
        res = pd.concat([s1.shift(i) for i in range(0, dimension + 1)], axis=1)
        res2 = res
        lin = len(res2)

        test = res2.iloc[lin-testNO:lin+1, 1:dimension + 1]
        test_target = res2.iloc[lin-testNO:lin+1, 0]

        l3=lin2-testNO



        tra = res2.iloc[dimension:int(numpy.floor(trainset * l3)), 1:dimension + 1]
        tra_target = res2.iloc[dimension:int(numpy.floor(trainset * l3)), 0]
        tra=tra.dropna()
        
        init_val1=int(numpy.floor(trainset * l3))
        end_val1 = int(numpy.floor((trainset+valset) * l3))
        init_val2 = int(numpy.floor((trainset+valset) * l3))
        end_val2=l3
        
        val = res2.iloc[init_val1:end_val1, 1:dimension + 1]
        val_target = res2.iloc[init_val1:end_val1, 0]


        val2 = res2.iloc[init_val2:end_val2, 1:dimension + 1]
        val_target2 = res2.iloc[init_val2:end_val2, 0]
        
        #test = res2.iloc[int(numpy.floor((trainset + valset) * lin)):lin + 1, 1:dimension + 1]
        #test_target = res2.iloc[int(numpy.floor((trainset + valset) * lin)):lin + 1, 0]


        lintra=len(tra_target)
        linval=len(val_target)
        lintest=len(test_target)
     #   tra = res2.iloc[dimension:int(numpy.floor(trainset * lin)), 1:dimension + 1]
        arima_train = data[0:int(numpy.floor(trainset * l3)) ]
        arima_val = data[init_val1 :end_val1]
        arima_val2 = data[init_val2 :end_val2]
        arima_test = data[l3:lin+1]
       # arima_train=tra_target
       # arima_val=val_target
        #arima_test=test_target
        return tra.values.tolist(), tra_target.values.tolist(), val.values.tolist(), val_target.values.tolist(),val2.values.tolist(),val_target2.values.tolist(), test.values.tolist(), test_target.values.tolist(), arima_train.tolist(), arima_val.tolist(),arima_val2.tolist(), arima_test.tolist()
