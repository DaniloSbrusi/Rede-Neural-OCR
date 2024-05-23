from leitor import MnistDataloader
from os.path import join
import random, math
import numpy as np

input_path = './archive/' 
training_images_filepath = join(input_path, 'train-images-idx3-ubyte', 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte', 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

NEURONIOS_OCULTOS = 100
TAXA = 0.8

saidas_desejadas = [] 
rede_neural = [[],[],[]] 
pesos = [] 
limiares = []

def configura_pesos():
    global pesos
    pesos = [[], []]
    pesos[0] = np.random.normal(scale=0.5, size=(28*28, NEURONIOS_OCULTOS))   
    pesos[1] = np.random.normal(scale=0.5, size=(NEURONIOS_OCULTOS, 10))   

def configura_limiares():
    global limiares
    limiares = [[], []]
    limiares[0] = np.random.normal(scale=0.5, size=NEURONIOS_OCULTOS)
    limiares[1] = np.random.normal(scale=0.5, size=10)
  
def configura_entradas(amostra):
    global rede_neural
    rede_neural[0] = np.array(amostra).reshape(-1) / 254

def configura_saidas(amostra):
    global saidas_desejadas
    saidas_desejadas = [0] * 10  
    saidas_desejadas[y_train[amostra]] = 1

def funcao_ativacao(x):
    return 1/(1+math.e**(-x)) # Sigmoid

def dsig(y):
    return y*(1.0-y)

def calcula_camadas():
    global rede_neural
    rede_neural[1] = funcao_ativacao(np.dot(rede_neural[0], pesos[0]) + limiares[0])
    rede_neural[2] = funcao_ativacao(np.dot(rede_neural[1], pesos[1]) + limiares[1])

def ajusta_parametros():
    out = rede_neural[2]
    erros_pesos_saida = rede_neural[1].reshape(-1,1)*(-(saidas_desejadas-out)*out*(1-out)).reshape(1,-1)*TAXA
    pesos[0] -= np.outer((np.sum(-(saidas_desejadas-out)*out*(1-out)*pesos[1],axis=1)*rede_neural[1]*(1-rede_neural[1])),rede_neural[0]).T
    pesos[1] -= erros_pesos_saida

def calcula_erro_total():
    erro = 0
    for saida in range(10):
        erro+=0.5*((saidas_desejadas[saida]-rede_neural[2][saida])**2)
    return erro

def imprime_saidas_treinamento(num_amostra):
    saida_arredondada = [round(num, 2) for num in rede_neural[2]]
    print('Amostra ' + str(num_amostra) + ' = Num. ' + str(y_train[num_amostra]) + ' --> ' + str(saida_arredondada) + '\t (Erro = ' + str(round(calcula_erro_total(),2)) + ')')
    
def treinamento():
    configura_pesos()
    configura_limiares()
    for num_amostra in range(len(x_train)):
        configura_entradas(x_train[num_amostra])
        configura_saidas(num_amostra)
        calcula_camadas()
        ajusta_parametros()
        # imprime_saidas_treinamento(num_amostra)

def teste():
    qtd_acertos = 0
    for num_amostra in range(len(x_test)):
        configura_entradas(x_test[num_amostra])
        calcula_camadas()
        # print(str(y_test[num_amostra]) + ' --> ' + str(np.argmax(rede_neural[2])))
        if(y_test[num_amostra] == np.argmax(rede_neural[2])):
            qtd_acertos += 1
    print(f"\nA quantidade de acertos foi {qtd_acertos} ({qtd_acertos*100/10000}%)\n")
    
treinamento()
teste()


