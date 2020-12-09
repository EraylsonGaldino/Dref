import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def create_windows(serie, n_in=3,n_out=1, dropnan=True):
    import pandas as pd

    serie = pd.DataFrame(serie)
    n_vars = 1  if type(serie) is list else serie.shape[1]
    df = pd.DataFrame(serie)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    for i in range(0, n_out):
        cols.append(df.shift(-i))

    agg = pd.concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.get_values()


def carregar_serie(endereco):    
    dados = pd.read_table(endereco, delimiter=',', header=None)
    return dados

def gerar_conv(neuronios, lags, filters = 4, kernel_size=2, pool_size= 1, func_opt='adam'):
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Flatten
	from keras.layers.convolutional import Conv1D
	from keras.layers.convolutional import MaxPooling1D
	model = Sequential()
	model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',   input_shape=(lags, 1)))
	model.add(MaxPooling1D(pool_size=pool_size))
	model.add(Flatten())
	model.add(Dense(neuronios, activation='relu'))
	
	model.add(Dense(1))
	model.compile(loss='mse', optimizer=func_opt)
		
	return model

def train_conv(modelo, x_train, y_train, x_val, y_val, num_ex=5,epochs=100):
	#[samples, timesteps, rows, cols, channels]
    from sklearn.metrics import mean_squared_error as MSE
    trainX = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    #trainX = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    valX =  x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
    #x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    melhor_mse = np.Inf
    for i in range(0, num_ex):
        
        modelo.fit(trainX, y_train, epochs=epochs, batch_size=1, verbose=0)
        prev_v = modelo.predict(valX)
        novo_mse  = MSE(y_val, prev_v)
        if novo_mse < melhor_mse:
            melhor_mse = novo_mse
            melhor_modelo = modelo
        
    return melhor_modelo, melhor_mse

def prev_conv(modelo, x_test):
    testX = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
	
    y_prev = modelo.predict(testX)
    previsao = []
    for i in range(0, len(y_prev)):
        previsao.append(y_prev[i][0])
	
	
    return previsao


def train_kfold(train_lags, lags):

    kfold = KFold(2, True, 1)
            
    neuronios = [5, 10, 100, 500] 
    func_opt = ['adam']
    best_result = np.Inf

    for i in range(0,len(neuronios)):
        for j in range(0,len(func_opt)):
            conv = gerar_conv(neuronios[i], lags, func_opt = func_opt[j])
            
            mse_folds = 0
            for train, test in kfold.split(train_lags):
                print('train: %s, test: %s' % (train_lags[train].shape, train_lags[test].shape))
                x_train = train_lags[train,0:lags]
                y_train = train_lags[train,-1]
                x_val = train_lags[test, 0:lags]
                y_val = train_lags[test, -1]              


                lstm, mse_val = train_conv(conv, x_train, y_train, x_val, y_val, num_ex=3,epochs=100)
                
                mse_folds += mse_val
                
            mse_val = mse_folds/2
                
            if mse_val < best_result:
                best_result = mse_val
                select_model = lstm
                print('melhor configuração. neuronios:', neuronios[i], 'funcao:', func_opt[j])
    
    return select_model




if __name__ == '__main__':
    series = ['red', 'car', 'elec', 'gas', 'lake', 'pigs', 'poll', 'suns', 'nordic', 'b1h']
    lags = 12

    
    for serie_name in series:
        if serie_name == 'b1h':
            lags = 24

        if serie_name == 'suns':
            lags = 11

        else:
            lags = 12

        print('executando série: ', serie_name)

        endereco_res = 'series/'+serie_name+'_res.txt'
        dados = carregar_serie(endereco_res)
        serie = dados[0].values

        train_lags = create_windows(serie,n_in= lags)

        x_train = train_lags[:,0:lags]
        y_train = train_lags[:,-1]

        print(train_lags.shape)

        #treinando o modelo 
        select_model = train_kfold(train_lags, lags)
        print(select_model)

        prev_tr = prev_conv(select_model, x_train)

        #carregando a serie de val 1
        endereco_res_val = 'series/'+serie_name+'_val.txt'
        dados = carregar_serie(endereco_res_val)
        serie_val = dados[0].values

        #junta com treinamento
        serie_val = np.hstack([serie[-lags:], serie_val])

        val_lags = create_windows(serie_val,n_in= lags)

        x_val = val_lags[:,0:lags]
        y_val = val_lags[:,-1]
        prev_val = prev_conv(select_model, x_val)

        #carregando a serie de val 2
        endereco_res_val_2 = 'series/'+serie_name+'_val_2.txt'
        dados = carregar_serie(endereco_res_val_2)
        serie_val_2 = dados[0].values

        #junta com os ultimos valores de val 1
        serie_val_2 = np.hstack([serie_val[-lags:], serie_val_2])

        val_lags_2 = create_windows(serie_val_2,n_in= lags)
        x_val_2 = val_lags_2[:,0:lags]
        y_val_2 = val_lags_2[:,-1]

        prev_val_2 = prev_conv(select_model, x_val_2)

        #carrega a serie de teste
        endereco_res_val_2 = 'series/'+serie_name+'_test.txt'
        dados = carregar_serie(endereco_res_val_2)
        serie_test = dados[0].values

        #junta com os ultimos valores de val 2
        serie_test = np.hstack([serie_val_2[-lags:], serie_test])

        test_lags = create_windows(serie_test,n_in= lags)
        x_test = test_lags[:,0:lags]
        y_test = test_lags[:,-1]  


        prev_test = prev_conv(select_model, x_test)

        #organiza as previsoes em DF 
        previsoes_tr = pd.DataFrame([y_train, prev_tr])
        previsoes_val_1 = pd.DataFrame([y_val, prev_val])
        previsoes_val_2 = pd.DataFrame([y_val_2, prev_val_2])
        previsoes_tes = pd.DataFrame([y_test, prev_test])

        previsoes_tr = previsoes_tr.transpose()
        previsoes_val_1 = previsoes_val_1.transpose()
        previsoes_val_2 = previsoes_val_2.transpose()
        previsoes_tes = previsoes_tes.transpose()

        previsoes_tr.columns = ['Target', 'Prev']
        previsoes_val_1.columns = ['Target', 'Prev']
        previsoes_val_2.columns = ['Target', 'Prev']
        previsoes_tes.columns = ['Target', 'Prev']

        #salva em xlsx
        name_save_result = serie_name+'_prev_conv.xlsx'
        writer = pd.ExcelWriter(name_save_result)
        previsoes_tr.to_excel(writer, 'Treinamento')
        previsoes_val_1.to_excel(writer, 'Validação 1')
        previsoes_val_2.to_excel(writer, 'Validação 2')
        previsoes_tes.to_excel(writer, 'Teste')
        writer.save()