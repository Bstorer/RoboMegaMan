import sys,os
sys.path.insert(0, 'evoman')
from environment import Environment
import datetime
import time
import numpy as np
import pickle
from funcoesProjeto import Controller,createModel,save_object



time1 =  datetime.datetime.now()#horario de inicio
start = time.time()#time inicial

experiment_name = "ProjetoIa"#nomeprojeto

somafs = []#vetor que guardara pontuacoes de cada geracao
controles = []#vetor que guardara os 10 controles de cada geracao
n = 10#numero de individuos por geracao 

init = input("Controles iniciais devem ser restaurados?")#se sim o codigo usa controles guardados na pasta controles do ultimo teste para iniciar primeira geracao. Se nao cria 10 individuos aleatorios


robos = [7,2,3,8]#ordem que chefoes serao enfrentados
if init == "sim":#define se como primeira geracao sera inciada
  with open('controles/ultRobo.txt', 'rb') as myfile:
    primeirorobo = int(myfile.read())#seleciona ultimo chefao com que esses controles lidaram
  for i in range(0,n):
    with open('controles/controle' + str(i) + '.pkl', 'rb') as inp:#le 10 controles guardados na pasta controles
      controle = pickle.load(inp)
      controles.insert(i,controle)
else:
  primeirorobo = robos[0]#primeiro robo sera o primeiro da ordem
  for i in range(0,n):#cria 10 individuos aleatorios
    model = createModel()
    controles.insert(i,Controller(model,np.random.randint(1000)))#controles possuem uma rede neural como atributo e um numero aleatorio entre 1 e 1000 indentificado como seu id

naoAchado = True
limite= 3600#limite de tempo q codigo roda eh 3600 segundos
for robo in robos:#para cada chefao fara o loop

  if naoAchado and robo!=primeirorobo:#caso os controles tenham sido retirados da pasta controle entao o primeiro chefao sera o ultimo q eles enfrentaram
    continue
  else:
    naoAchado = False

  if time.time() - start >= limite:#se limite de tempo for alcancado loop para
    break

  totalFit = 0

  while True:#o loop ocorrera de geracao em geracao

    if time.time() - start >= limite:#se limite de tempo for alcancado loop para
      print("Ultimo robo em que treinou = ",robo)
      ultRobo = robo#eh gravado ultimo robo que a geracao atual enfrentou
      break


    fitnesses = []#vetor que guardara os fitnesses de cada individuo da geracao atual

    for i in range(0,n): #avaliara o desempenho de cada controle da geracao atual
      print("agentes disponiveis:")
      for controle in controles:#printa na tela os ids dos controles disponivels (apenas um artificio visual)
        print("->  ",controle.n)
      print("testando agente numero: ",controles[i].n)#indica qual o controle atual que tera fitness recolhido

      env = Environment(experiment_name=experiment_name,#gera environment em que controle sera colocado a prova
              playermode="ai",
              enemies=[robo],
              player_controller=controles[i],
              randomini="yes",
              speed="fastest",
              contacthurt ="player",
              level=2)

      env.play()#inicia jogo
      fitnesses.insert(i,env.fitness_single())#coleta fit que controle conseguiu atingir no jogo
      print("FITNESS - ",fitnesses," do agente ", controles[i].n)# printa o fit e o controle na tela

    fpais = []#vetor que guardara fitnesses dos controles de maior desempenho da geracao atual
    indexes = []#vertor que guardara indices dos controles de maior desempenho atual no vetor controles
    somafs = somafs + [sum(fitnesses)]#guarda pontuacao total da geracao atual no vetor somafs
    totalFit = sum(fitnesses)#guarda a pontuacao total da geracao na variavel totalFit

    print("TOTAL FIT da geracao ",totalFit)#printa o total fit da geracao atual na tela

    fitAux = []#fitAux eh apenas uma variavel auxiliar para guardar o vetor de fitnesses da geracao atual, pois o vetor de fitnesses sera alterado durante a fase de selecao
    fitAux += fitnesses
    
    if totalFit  > 800:#se geracao atua atingiu uma pontuacao total maior que 800 significa que ela esta aproximadamente treinada com o chefe atual e pode enfrentar um novo
      ultRobo = robo#salva o numero do ultimo chefe a ser enfrentado 
      break

    for j in range(0,int(n/2)):#seleciona n/2 melhores individuos da geracao

      fpai = max(fitnesses)#seleciona o individuo de maior desenpenho no vetor
      index = fitnesses.index(fpai)#seleciona o indice do individuo de maior desempenho no vetor

      fitnesses[index] = -100000#negativa a pontuacao do individuo de maior desempenho no vetor pois ele ja foi escolhido


      fpais.insert(j,fpai)#salva pontuacao do individuo de maior desempenho do vetor
      indexes.insert(j,index)#salva o indice do individuo de maior desempenho do vetor

    for c in indexes:#printa na tela os ids dos controles de melhro desempenho da geracao
      print("escolhido -> ", controles[c].n)

    filhos = []#vetr que guardara filhos da geracao atual
    for j in range(0,int(n/2)):#fase de reproducao gerando n/2 filhos

      while True:#escolhe dois individuos distintos dentre os n/2 individuos de maior desempenho d ageracao atual
        pai1 = np.random.randint(int(n/2))
        pai2 = np.random.randint(int(n/2))
        index1 = indexes[pai1]
        index2 = indexes[pai2]
        if index1 != index2:
          break

      chance1 = np.random.random()
      controle1 = controles[index1] if chance1 <= 0.5 else controles[np.random.randint(n)]#ha 50% de chance do pai 1 ser exclusivamente um dos mehores n/2 individuos da geracao atual e 50% de chance de ser qualquer controle da geracao atual
      chance2 = np.random.random()
      controle2 = controles[index2] if chance2 <= 0.5 else controles[np.random.randint(n)]#ha 50% de chance do pai 1 ser exclusivamente um dos mehores n/2 individuos da geracao atual e 50% de chance de ser qualquer controle da geracao atual

      print("juntando ",controle1.n,controle2.n)
      a = np.random.random()
      b = 1 - a 
      pesos1 = np.array(controle1.model.get_weights())  
      pesos2 = np.array(controle2.model.get_weights()) 

      pesosFilho = a*pesos1 + b*pesos2#cruza os coeficientes dos dois pais atraves dos fatores aleatorios a e b


      modelofilho = createModel(pesosFilho)#cria rede neural com os pesos gerados
      filho = Controller(modelofilho,controle1.n + controle2.n)#cria novo objeto controler
      filhos.insert(j,filho)#insere dentro do vetor que guardara filhos

    z = 0
    for j in range(n):#substitui piores pais por filhos (tem uma chance desses pais ns erem substituidos)
      perc = np.random.random()
      if j not in indexes:
        if perc <= 0.6 and z < len(filhos):#os n/2 piores individuos da geracao atual de 60% de chance de serem substituidos por um filho da geracao atual
          controles[j] = filhos[z]
          z += 1
      else:
        if perc <= 0.1 and z < len(filhos):#os n/2 melhores individuos da geracao atual de 10% de chance de serem substituidos por um filho da geracao atual
          controles[j] = filhos[z]
          z += 1

    for j in range(n):#todo filho que entra para a proxima geracao tem 40% de chance de sofrer ma mutacao aleatoria
      perc = np.random.random()
      if perc <= 0.4 and controles[j] in filhos:
        print("mutacao em ",controles[j].n)
        pesos = np.array(controles[j].model.get_weights())
        gaussianL1 = np.random.normal(size = 260).reshape((20,13))
        gaussianL2 = np.random.normal(size = 65).reshape((13,5))
        pesos[0] = pesos[0] + gaussianL1
        pesos[2] = pesos[2] + gaussianL2
        controles[j].model.set_weights(pesos) 
    





m = 0
for c in controles:#salva todo controle da geracao atual em um arquivo dentro da pasta controles
  save_object(c, 'controles/controle' + str(m) + '.pkl')
  m += 1

print("fitAux: ",fitAux)#fit da ultima geracao
f = max(fitAux)
index = fitAux.index(f)
controleFinal = controles[index]
print("controleFinal: ",controleFinal.n)
save_object(controleFinal, 'controles/controleFinal.pkl')#salva o melhor controle da geracao atual em um arquivo especial que o evidencia
with open('controles/ultRobo.txt', 'r+') as myfile:
  myfile.write(str(ultRobo))


time2 = datetime.datetime.now()#horario de finalizacao do treino
print("iniciado em ", time1, " terminado em ",time2)

