import sys,os
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np
import pickle
from funcoesProjeto import Controller,createModel,save_object


experiment_name = "ProjetoIa"

with open('agenteFinal.pkl', 'rb') as inp:#seleciona agenteFinal
    controle = pickle.load(inp)
    print("controle: ",controle.n)


while True:
  for robo in [1,2,3,4,5,6,7,8]:#coloca o agente contra todos os chefes
    env = Environment(experiment_name=experiment_name,
                playermode="ai",
                enemies=[robo],
                randomini="yes",
                player_controller=controle,
                speed="normal",
                contacthurt ="player",
                level=2)
    env.play()
    print("FITNESS - ",env.fitness_single())