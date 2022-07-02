
import pandas as pd
import numpy as np

def calc_return(df, hp):
    '''Calcula o retorno de uma série de preços'''
    df_returns = df/df.shift(hp) -1
    df_returns = df_returns.dropna()
    return df_returns


def read_infomoney(file_name):
    '''Lê e trata um csv/txt'''
    df = pd.read_csv('data/'+file_name, sep=',', decimal=',')
    df = df[['DATA', 'FECHAMENTO']]
    df = df.set_index('DATA')
    return df

# otimização de portfólio: mínimo risco para determinado nível de retorno
def f_obj_min_risk(w,cov_matrix):
    '''Recebe um vetor de pesos de ativos na carteira e a mátrix de covariância entre os retornos
       Retorna o desvio padrão padrão da carteira    
    '''
    return np.sqrt(np.dot(w, np.dot(cov_matrix, w))) # definindo função-objetivo: minimizar risco

# otimização de portfólio: máximo retorno para determinado nível de risco
def f_obj_max_ret(w,mi):
    '''Recebe o vetor de pesos de ativos na carteira e o vetor de média dos retornos da carteira
       Retorna o retorno esperado
    '''
    return -np.sum(w*mi) # definindo função-objetivo: maximizar retorno