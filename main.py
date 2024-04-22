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
TAXA = 0.3

entradas = []
saidas_desejadas = [] 
rede_neural = [[],[],[]] 
pesos = [] 
limiares = []


def inicializacao_xavier_pesos(n_entrada, n_saida):
    limite = np.sqrt(6.0 / (n_entrada + n_saida))
    pesos = np.random.uniform(-limite, limite, size=(n_entrada, n_saida))
    return pesos

def configura_pesos():
    global pesos
    pesos = [np.array([]), np.array([])]
    pesos[0] = inicializacao_xavier_pesos(28*28, NEURONIOS_OCULTOS)
    pesos[1] = inicializacao_xavier_pesos(NEURONIOS_OCULTOS, 10)

def inicializacao_xavier_limiares(num_neuronios_entrada, num_neuronios_saida):
    limite = math.sqrt(1.0 / num_neuronios_entrada)
    limiares = [random.uniform(-limite, limite) for _ in range(num_neuronios_saida)]
    return limiares

def configura_limiares():
    global limiares
    limiares = [[], []]
    limiares[0] = inicializacao_xavier_limiares(28*28, NEURONIOS_OCULTOS)
    limiares[1] = inicializacao_xavier_limiares(NEURONIOS_OCULTOS, 10)
  
def configura_entradas(amostra):
    global rede_neural
    rede_neural[0] = np.array(x_train[amostra]).reshape(-1) / 254

def configura_saidas(amostra):
    global saidas_desejadas
    saidas_desejadas = [0] * 10  
    saidas_desejadas[y_train[amostra]] = 1

def funcao_ativacao(x):
    return 1/(1+math.e**(-x)) # Sigmoid

def calcula_camadas():
    global rede_neural
    rede_neural[1] = funcao_ativacao(np.dot(rede_neural[0], pesos[0]) + limiares[0])
    rede_neural[2] = funcao_ativacao(np.dot(rede_neural[1], pesos[1]) + limiares[1])

def ajusta_parametros():

    global pesos, limiares
    pesos_novos = pesos

    for neuronio_saida in range(10):
        delta_saida = (rede_neural[2][neuronio_saida]-saidas_desejadas[neuronio_saida])*rede_neural[2][neuronio_saida]*(1-rede_neural[2][neuronio_saida])
        pesos_novos[1][:, neuronio_saida] -= TAXA*delta_saida*rede_neural[1]
        limiares[1][neuronio_saida] -= TAXA*delta_saida

    for neuronio_saida in range(10):
        delta_saida = (rede_neural[2][neuronio_saida] - saidas_desejadas[neuronio_saida]) * rede_neural[2][neuronio_saida] * (1 - rede_neural[2][neuronio_saida])
        delta_oculto = rede_neural[1] * (1 - rede_neural[1]) * delta_saida * pesos[1][:, neuronio_saida]

        for neuronio_entrada in range(28*28):
            delta = np.dot(delta_oculto, pesos[0][neuronio_entrada, :]) * rede_neural[0][neuronio_entrada]
            pesos_novos[0][neuronio_entrada, :] -= TAXA * delta

        limiares[0][neuronio_saida] -= TAXA * np.sum(delta * delta_oculto)

    pesos = pesos_novos

def calcula_erro_total():
    erro = 0
    for saida in range(10):
        erro+=0.5*((saidas_desejadas[saida]-rede_neural[2][saida])**2)
    return erro

def treinamento():
    configura_pesos()
    configura_limiares()
    for num_amostra in range(len(x_train)):
        configura_entradas(num_amostra)
        configura_saidas(num_amostra)
        calcula_camadas()
        ajusta_parametros()
        # saida_arredondada = [round(num, 2) for num in rede_neural[2]]
        # print('Num. ' + str(y_train[num_amostra]) + ' --> ' + str(saida_arredondada) + ' (Erro = ' + str(round(calcula_erro_total(),2)) + ')')
        print(calcula_erro_total())

treinamento()
