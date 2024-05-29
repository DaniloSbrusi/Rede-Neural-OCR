from leitor import MnistDataloader
from os.path import join
import math, numpy as np

class RedeNeural:

    def __init__(self, input_path='./archive/', neuronios_ocultos=100, taxa=0.8):
        self.input_path = input_path
        self.neuronios_ocultos = neuronios_ocultos
        self.taxa = taxa
        self.saidas_desejadas = []
        self.rede_neural = [[], [], []]
        self.pesos = []
        self.limiares = []

        self.training_images_filepath = join(input_path, 'train-images-idx3-ubyte', 'train-images-idx3-ubyte')
        self.training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte', 'train-labels-idx1-ubyte')
        self.test_images_filepath = join(input_path, 't10k-images-idx3-ubyte', 't10k-images-idx3-ubyte')
        self.test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')

        self.mnist_dataloader = MnistDataloader(self.training_images_filepath, self.training_labels_filepath, self.test_images_filepath, self.test_labels_filepath)

        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist_dataloader.load_data()

    def configura_pesos(self):
        self.pesos = [
            np.random.normal(scale=0.5, size=(28*28, self.neuronios_ocultos)),
            np.random.normal(scale=0.5, size=(self.neuronios_ocultos, 10))
        ]

    def configura_limiares(self):
        self.limiares = [
            np.random.normal(scale=0.5, size=self.neuronios_ocultos),
            np.random.normal(scale=0.5, size=10)
        ]
    
    def configura_entradas(self, imagem):
        self.rede_neural[0] = np.array(imagem).reshape(-1) / 254

    def configura_saidas_desejadas(self, amostra):
        self.saidas_desejadas = [0] * 10  
        self.saidas_desejadas[self.y_train[amostra]] = 1

    def funcao_ativacao(self, x):
        return 1/(1+math.e**(-x))

    def calcula_camadas(self):
        self.rede_neural[1] = self.funcao_ativacao(np.dot(self.rede_neural[0], self.pesos[0]) + self.limiares[0])
        self.rede_neural[2] = self.funcao_ativacao(np.dot(self.rede_neural[1], self.pesos[1]) + self.limiares[1])

    def ajusta_parametros(self):
        out = self.rede_neural[2]
        erros_pesos_saida = self.rede_neural[1].reshape(-1,1)*(-(self.saidas_desejadas-out)*out*(1-out)).reshape(1,-1)*self.taxa
        self.pesos[0] -= np.outer((np.sum(-(self.saidas_desejadas-out)*out*(1-out)*self.pesos[1],axis=1)*self.rede_neural[1]*(1-self.rede_neural[1])),self.rede_neural[0]).T
        self.pesos[1] -= erros_pesos_saida

    def calcula_erro_total(self):
        erro = 0
        for saida in range(10):
            erro+=0.5*((self.saidas_desejadas[saida]-self.rede_neural[2][saida])**2)
        return erro

    def imprime_saidas_treinamento(self, num_amostra):
        saida_arredondada = [round(num, 2) for num in self.rede_neural[2]]
        print('Amostra ' + str(num_amostra) + ' = Num. ' + str(self.y_train[num_amostra]) + ' --> ' + str(saida_arredondada) + '\t (Erro = ' + str(round(self.calcula_erro_total(),2)) + ')')
        
    def treinamento(self):
        self.configura_pesos()
        self.configura_limiares()
        for num_amostra in range(len(self.x_train)):
            self.configura_entradas(self.x_train[num_amostra])
            self.configura_saidas_desejadas(num_amostra)
            self.calcula_camadas()
            self.ajusta_parametros()
            # imprime_saidas_treinamento(num_amostra)

    def teste(self):
        qtd_acertos = 0
        for num_imagem in range(len(self.x_test)):
            self.configura_entradas(self.x_test[num_imagem])
            self.calcula_camadas()
            # print(str(y_test[num_imagem]) + ' --> ' + str(np.argmax(rede_neural[2])))
            if(self.y_test[num_imagem] == np.argmax(self.rede_neural[2])):
                qtd_acertos += 1
        print(f"\nA quantidade de acertos foi {qtd_acertos} ({qtd_acertos*100/10000}%)\n")

    def processamento(self, imagem):
        self.configura_entradas(imagem)
        self.calcula_camadas
        return np.argmax(self.rede_neural[2])

if __name__ == "__main__":
    rede_neural = RedeNeural()
    rede_neural.treinamento()
    rede_neural.teste()
