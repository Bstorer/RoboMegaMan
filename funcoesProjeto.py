from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import numpy as np
import pickle

def findFit(env):#adquire o valor de fit do agente
  pv = env.get_playerlife()
  ev = env.get_enemylife()
  t = env.get_time()
  if pv <= 0:
    fit = (100 - ev)*(t**2)
  else:
    fit = ((100 - ev)*(pv**2) + math.log(t,10))
  return fit

def createModel(pesos = None):#cria rede neural com pesos especificos ou aleatórios

  model = Sequential()
  model.add(Dense(13, input_dim=20, activation='relu'))
  model.add(Dense(5, activation='softmax'))
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

  if pesos != None:
    model.set_weights(pesos) 

  return model

def save_object(obj, filename):#salva modelo 
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

class Controller(object):#da play pra kd controller e compara resultados
  def __init__(self,model,n):
    self.n = n#id gerado aleatoriamente que indentifica numero do controle (apenas para questao de layout)
    self.model = model
  def control(self, params, cont = None):#params sao os params do jogo
    modelo = self.model
    entrada = np.array(params).reshape((1,20))
    acoes = modelo.predict(entrada).tolist()[0]
    left = (1 if acoes[0] >= 0.5 else 0) #andar esquerda
    right = (1 if acoes[1] >= 0.5 else 0)#andar direita
    jump = (1 if acoes[2] >= 0.5 else 0)#pular
    shoot = (1 if acoes[3] >= 0.5 else 0)#atirar
    release = (1 if acoes[4] >= 0.5 else 0)#desistir do pular
    return [left, right, jump, shoot, release] #acoes q jogo fara


    