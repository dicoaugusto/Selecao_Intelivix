from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
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



###############MLPClassifier##############


redeneural=MLPClassifier(verbose=True,
                         max_iter=1000,
                         tol=0.00001,
                         activation='tanh',
                         learning_rate_init=0.001)

#redeneural.fit(entradas,saidas)


redeneural.fit(entradaTreinamento,saidaTreinamento)


predicao=redeneural.predict(entradaTeste)

predicao=predicao.astype('int32')
#predicao=predicao.reshape(1,-1)

saidaTeste=saidaTeste.astype('int32')
#saidaTeste=saidaTeste.reshape(1,-1)

#calcular acuracia
linha=0
coluna=0
tamanho=len(predicao)
igual=''
igual=0
for linha in range(tamanho):
    if predicao[linha]==saidaTeste[linha]:
        igual=igual+1

acuracia=igual/tamanho
print(acuracia)
        

#redeneural.score(predicao,saidaTeste)
