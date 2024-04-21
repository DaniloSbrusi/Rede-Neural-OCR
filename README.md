# Rede Neural OCR

Implementação de uma rede neural simples para reconhecimento de dígitos em 3 camadas :

- Na primeira camada, de entrada, são inseridos os 784 pixels de cada imagem
- Na segunda camada, dita oculta, o número de neurônios é configurável
- Na terceira camada, de saída, há 10 neurônios que representam cada um dos dígitos (0-9)

A rede neural ainda possui pesos para cada conexão entre os neurônios de suas camadas e valores de limiar individualizados por neurônio

---

Conjunto de dados MNIST de imagens de dígitos manuscritos com 28x28 pixels já tratadas em escala de cinza
- 60000 imagens para treino
- 10000 imagens para teste

Obtido em https://www.kaggle.com/datasets/hojjatk/mnist-dataset

