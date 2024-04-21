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

NEURONIOS_CAMADA_OCULTA = 100
TAXA_APRENDIZADO = 0.7

entradas = []
saidas_desejadas = [] 
rede_neural = [[],[],[]] # 3 camadas de neurônios
pesos = [] # Pesos das conexões entre neurônios
limiares = [] # Bias dos neurônios da camada oculta e saída


def inicializacao_xavier_pesos(n_entrada, n_saida):
    limite = np.sqrt(6.0 / (n_entrada + n_saida))
    pesos = np.random.uniform(-limite, limite, size=(n_entrada, n_saida))
    return pesos

def configura_pesos_xavier():
    global pesos
    pesos = [{},{}]
    pesos[0] = inicializacao_xavier_pesos(28*28, NEURONIOS_CAMADA_OCULTA)
    pesos[1] = inicializacao_xavier_pesos(NEURONIOS_CAMADA_OCULTA, 10)

def configura_pesos_aleatorios():
    global pesos
    pesos = [{},{}]
    for neuronio_entrada in range(28*28):
        for neuronio_oculto in range(NEURONIOS_CAMADA_OCULTA):
            pesos[0][(neuronio_entrada, neuronio_oculto)] = random.random()
    for neuronio_oculto in range(NEURONIOS_CAMADA_OCULTA):
        for neuronio_saida in range(10):
            pesos[1][(neuronio_oculto, neuronio_saida)] = random.random()
    
def inicializacao_xavier_pesos_limiares(num_neuronios_entrada, num_neuronios_saida):
    limite = math.sqrt(1.0 / num_neuronios_entrada)
    limiares = [random.uniform(-limite, limite) for _ in range(num_neuronios_saida)]
    return limiares

def configura_limiares_xavier():
    global limiares
    limiares = [[], []]
    limiares[0] = inicializacao_xavier_pesos_limiares(28*28, NEURONIOS_CAMADA_OCULTA)
    limiares[1] = inicializacao_xavier_pesos_limiares(NEURONIOS_CAMADA_OCULTA, 10)

def configura_limiares_aleatorios():
    global limiares
    limiares = [[],[]]
    for neuronio in range(NEURONIOS_CAMADA_OCULTA):
        limiares[0].append(random.random())
    for neuronio in range(10):
        limiares[1].append(random.random())
    
def configura_entradas(amostra):
    global rede_neural
    rede_neural[0].clear()
    for linha in range(28):
        for coluna in range(28):
            rede_neural[0].append(x_train[amostra][linha][coluna]/254) # Valores de entrada entre 0 e 1

def configura_saidas(amostra):
    global saidas_desejadas
    saidas_desejadas = [0] * 10  # Inicializa todas as saídas como 0
    saidas_desejadas[y_train[amostra]] = 1

def funcao_ativacao(x):
    return 1/(1+math.e**(-x)) # Sigmoid

def calcula_camadas():
    global rede_neural
    rede_neural[1].clear()
    rede_neural[2].clear()
    for neuronio_oculto in range(NEURONIOS_CAMADA_OCULTA):
        somatorio = 0
        for neuronio_entrada in range(28*28):
            somatorio += rede_neural[0][neuronio_entrada]*pesos[0][(neuronio_entrada, neuronio_oculto)]
        somatorio += limiares[0][neuronio_oculto]
        rede_neural[1].append(funcao_ativacao(somatorio))
    for neuronio_saida in range(10):
        somatorio = 0
        for neuronio_oculto in range(NEURONIOS_CAMADA_OCULTA):
            somatorio += rede_neural[1][neuronio_oculto]*pesos[1][(neuronio_oculto, neuronio_saida)]
        somatorio += limiares[0][neuronio_saida]
        rede_neural[2].append(funcao_ativacao(somatorio))

def ajusta_pesos_e_limiares():
    global pesos, limiares
    pesos_novos = [{},{}]
    limiares_novos = [[],[]]

    # Ajuste pesos para camada de saída
    for neuronio_saida in range(10):
        for neuronio_oculto in range(NEURONIOS_CAMADA_OCULTA):
            pesos_novos[1][(neuronio_oculto, neuronio_saida)] = pesos[1][(neuronio_oculto, neuronio_saida)]-(TAXA_APRENDIZADO*(rede_neural[2][neuronio_saida]-saidas_desejadas[neuronio_saida])*rede_neural[2][neuronio_saida]*(1-rede_neural[2][neuronio_saida])*rede_neural[1][neuronio_oculto])
        # Ajuste limiares da camada de saída
        limiares_novos[1].append(limiares[1][neuronio_saida]-(TAXA_APRENDIZADO*rede_neural[2][neuronio_saida]*(1-rede_neural[2][neuronio_saida])*(rede_neural[2][neuronio_saida]-saidas_desejadas[neuronio_saida])))

    # Ajuste de pesos para camada oculta
    for neuronio_oculto in range(NEURONIOS_CAMADA_OCULTA):
        for neuronio_entrada in range(28*28):
            delta = 0
            for neuronio_saida in range(10):
                delta+=(rede_neural[2][neuronio_saida]-saidas_desejadas[neuronio_saida])*rede_neural[2][neuronio_saida]*(1-rede_neural[2][neuronio_saida])*pesos[1][(neuronio_oculto, neuronio_saida)]
            pesos_novos[0][(neuronio_entrada, neuronio_oculto)] = pesos[0][(neuronio_entrada, neuronio_oculto)]- TAXA_APRENDIZADO*delta*rede_neural[1][neuronio_oculto]*(1-rede_neural[1][neuronio_oculto])*rede_neural[0][neuronio_entrada]
        # Ajuste de limiares da camada oculta
        delta = 0
        for neuronio_saida in range(10):
            delta += (rede_neural[2][neuronio_saida]-saidas_desejadas[neuronio_saida])*rede_neural[2][neuronio_saida]*(1-rede_neural[2][neuronio_saida])*pesos[1][(neuronio_oculto, neuronio_saida)]
        limiares_novos[0].append(limiares[0][neuronio_oculto]-TAXA_APRENDIZADO*delta*rede_neural[1][neuronio_oculto]*(1-rede_neural[1][neuronio_oculto]))

    # Atualiza valores
    limiares = limiares_novos
    pesos = pesos_novos

def calcula_erro_total():
    erro = 0
    for saida in range(10):
        erro+=0.5*((saidas_desejadas[saida]-rede_neural[2][saida])**2)
    return erro

def treinamento():
    for num_amostra in range(len(x_train)):
        configura_entradas(num_amostra)
        configura_saidas(num_amostra)
        calcula_camadas()
        ajusta_pesos_e_limiares()
        saida_arredondada = [round(num, 2) for num in rede_neural[2]]
        print('Num. ' + str(y_train[num_amostra]) + ' --> ' + str(saida_arredondada) + ' (Erro = ' + str(round(calcula_erro_total(),2)) + ')')

configura_pesos_xavier()
configura_limiares_xavier()
treinamento()
