import pandas as pd
import keras
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.layers import Dense #camadas densas na rede neural
import nltk

base_comentario_total = pd.read_table('train.tsv',usecols=[0,2,3])
#base_comentario_predicao = pd.read_table('train.tsv',usecols=[0,2])

base_comentarios_treinamento,base_comentarios_teste=train_test_split(base_comentario_total,test_size=0.3, random_state=42)

tamanho_treinamento=len(base_comentarios_treinamento)
tamanho_teste=len(base_comentarios_teste)

base_comentarios_treinamento_lista=[]
for i in range(0,tamanho_treinamento):
    base_comentarios_treinamento_lista.append([str(base_comentarios_treinamento.values[i,j]) for  j in range(1,3)])


base_comentarios_teste_lista=[]
for i in range(0,tamanho_teste):
    base_comentarios_teste_lista.append([str(base_comentarios_teste.values[i,j]) for  j in range(1,3)])



stopwordscompleto = nltk.corpus.stopwords.words('english')


def RetirarStopWordsRadical(texto):
    radical=nltk.stem.SnowballStemmer('english')
    frasesSemRadical = []
    for (Phrase,Sentiment) in texto:
        FraseLimpa=[str(radical.stem(p)) for p in Phrase.split()  if p not in stopwordscompleto]
        frasesSemRadical.append((FraseLimpa,Sentiment))
    return frasesSemRadical


FrasesTreinamentoSemRadical =RetirarStopWordsRadical(base_comentarios_treinamento_lista)
FrasestesteSemRadical =RetirarStopWordsRadical(base_comentarios_teste_lista)


###limitar a dimensionalidade a 10 palavras por linha


##############TREINAMENTO
linha=[]
linhacomdez=[]
linha2=[]
linhacomdez2=[]
for i in range(len(FrasesTreinamentoSemRadical)):
    linha=FrasesTreinamentoSemRadical[i][0]
    tamanho=len(linha)
    if tamanho==10:
        linhacomdez.append(linha)
        linha2=FrasesTreinamentoSemRadical[i][1]
        linhacomdez2.append(linha2)


###########################BASE DE TESTE######################
linhaTeste=[]
linhacomdezTeste=[]
linha2teste=[]
linhacomdez2teste=[]
for i in range(len(FrasestesteSemRadical)):
    linhaTeste=FrasestesteSemRadical[i][0]
    tamanho=len(linhaTeste)
    if tamanho==10:
        linhacomdezTeste.append(linhaTeste)
        linha2teste=FrasesTreinamentoSemRadical[i][1]
        linhacomdez2teste.append(linha2teste)


###array
entradaTreinamento=np.array(linhacomdez)
#entradaTreinamentoorigial=np.array(linhacomdez,dtype=str)
saidaTreinamento=np.array(linhacomdez2)
#entradaTreinamentoorigial=entradaTreinamento
entradaTeste=np.array(linhacomdezTeste)
#entradaTesteOriginal=np.array(linhacomdezTeste)
saidaTeste=np.array(linhacomdez2teste)



#trocando textos por números
linha=0
coluna=0
entradaDePara=np.array
numero=0
qtlinhasmatrix=len(entradaTreinamento)
qtlinhasmatrixTeste=len(entradaTeste)
for linha in range(qtlinhasmatrix):
    for coluna in range(10):
            numero=numero+1
            palavra=entradaTreinamento[linha,coluna]
            linhaloc=0
            colunaloc=0
            for linhaloc in range(qtlinhasmatrix): #localizando na matrix treinamento
                for colunaloc in range(10):
                    if entradaTreinamento[linhaloc,colunaloc]==palavra:
                        entradaTreinamento[linhaloc,colunaloc]=numero
                        linhamatrixteste=0
                        colunamatrixteste=0
                        for linhamatrixteste in range(qtlinhasmatrixTeste): #localizando na matrix teste
                            for colunamatrixteste in range(10):
                                if entradaTeste[linhamatrixteste,colunamatrixteste]==palavra:
                                    entradaTeste[linhamatrixteste,colunamatrixteste]=numero
                                    

########
#try:
#    print(int(entradaTeste[0,0]))
#except:
#    print('n')



                                

#para as palavras não localizadas na entrada teste
linha=0
coluna=0
numero=numero+1
inteiro=''
qtlinhasmatrixTeste=len(entradaTeste)
for linha in range(qtlinhasmatrixTeste):
    for coluna in range(10):
        numero=numero+1
        palavra=entradaTeste[linha,coluna]
        try:
            palavra=int(palavra)
            inteiro='S'
        except:
            inteiro='N'
        if inteiro=='N':
            linhamatrixteste=0
            colunamatrixteste=0
            for linhamatrixteste in range(qtlinhasmatrixTeste):
                for colunamatrixteste in range(10):
                    if entradaTeste[linhamatrixteste,colunamatrixteste]==palavra:
                        entradaTeste[linhamatrixteste,colunamatrixteste]=int(numero)
                        
                    


entradaTreinamento=entradaTreinamento.astype('int32')
entradaTeste=entradaTeste.astype('int32')

#################################################################################
###########################################MONTANDO A REDE DENSA#################
##############################################################################
classificador=Sequential()

###camada entrada / oculta
classificador.add(Dense(units=10,activation='relu'
                        ,kernel_initializer='random_uniform',input_dim=30))

###nova camada (sequencial)
classificador.add(Dense(units=10,activation='relu'
                        ,kernel_initializer='random_uniform'))


###camada saida
classificador.add(Dense(units=1,activation='sigmoid'))

###otimizando o gradiente com keras
otimizador=keras.optimizers.Adam(lr=0.001,decay=0.0001,clipvalue=0.5)


##descida gradiente estocastico(optimizer=adam)
classificador.compile(optimizer=otimizador,loss='binary_crossentropy'
                      ,metrics=['binary_accuracy'])

##atualização dos pesos a cada 10 registros(batch_size=10)

classificador.fit(entradaTreinamento,saidaTreinamento,
                  batch_size=10,epochs=100)


####verificação de pesos
pesos0=classificador.layers[0].get_weights()
print(len(pesos0))

pesos1=classificador.layers[1].get_weights()

pesos2=classificador.layers[2].get_weights()

previsoes=classificador.predict(entradaTeste)
previsoes=previsoes>0.5

###sklear
precisao=accuracy_score(saidaTeste,previsoes)
matriz=confusion_matrix(saidaTeste,previsoes)

##keras
#resultado=classificador.evaluate(previsores_teste,classe_teste)