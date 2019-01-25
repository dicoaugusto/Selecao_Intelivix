
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.metrics import ConfusionMatrix



base_comentario_total = pd.read_table('train.tsv',usecols=[0,2,3])


base_comentarios_treinamento,base_comentarios_teste=train_test_split(base_comentario_total,test_size=0.3, random_state=42)

tamanho_treinamento=len(base_comentarios_treinamento)
tamanho_teste=len(base_comentarios_teste)

base_comentarios_treinamento_lista=[]
for i in range(0,tamanho_treinamento):
    base_comentarios_treinamento_lista.append([str(base_comentarios_treinamento.values[i,j]) for  j in range(1,3)])


base_comentarios_teste_lista=[]
for i in range(0,tamanho_teste):
    base_comentarios_teste_lista.append([str(base_comentarios_teste.values[i,j]) for  j in range(1,3)])



base_comentario_predicao,base_comentarios_teste=train_test_split(base_comentarios_teste,test_size=0.5)

tamanho_predicao=len(base_comentario_predicao)


base_comentario_predicao_lista=[]
for i in range(0,tamanho_predicao):
    base_comentario_predicao_lista.append([str(base_comentario_predicao.values[i,j]) for j in range(0,2)])


#####REMOVENDO OS STOPWORDS  e OS RADICAIS

stopwordscompleto = nltk.corpus.stopwords.words('english')


def RetirarStopWordsRadical(texto):
    radical=nltk.stem.SnowballStemmer('english')
    frasesSemRadical = []
    for (Phrase,Sentiment) in texto:
        FraseLimpa=[str(radical.stem(p)) for p in Phrase.split()  if p not in stopwordscompleto]
#        FraseLimpa=[str(radical.stem(p)) for p in Phrase.split()]
        frasesSemRadical.append((FraseLimpa,Sentiment))
    return frasesSemRadical






FrasesTreinamentoSemRadical =RetirarStopWordsRadical(base_comentarios_treinamento_lista)
FrasestesteSemRadical =RetirarStopWordsRadical(base_comentarios_teste_lista)



######PALAVRAS UNICAS
def buscapalavras(frases):
    todaspalavras=[]
    for (Phrase,Sentiment) in frases:
        todaspalavras.extend(Phrase)
    return todaspalavras





palavrasTreinamento = buscapalavras(FrasesTreinamentoSemRadical)
palavrasTeste = buscapalavras(FrasestesteSemRadical)

#####CONTABILIZANDO A FREQUENCIA DOS RADICAIS
def buscafrequencia(palavras):
    palavras=nltk.FreqDist(palavras)
    return palavras


frequenciaTreinamento = buscafrequencia(palavrasTreinamento)
frequenciaTeste = buscafrequencia(palavrasTeste)

#############PALAVRAS UNICAS
def buscapalavrasunicas(frequencia):
    freq=frequencia.keys()
    return freq

palavrasunicasTreinamento=buscapalavrasunicas(frequenciaTreinamento)
palavrasunicasTeste=buscapalavrasunicas(frequenciaTeste)


###RETORNAR EXTRATOR DE PALAVRAS
def extratorpalavrasTreinamento(documento):
    doc = set(documento)
    caracteristicasTreinamento={}
    for palavras in palavrasunicasTreinamento:
        caracteristicasTreinamento['%s' % palavras]=(palavras in doc)
    return caracteristicasTreinamento


def extratorpalavrasTeste(documento):
    doc = set(documento)
    caracteristicasTeste={}
    for palavras in palavrasunicasTeste:
        caracteristicasTeste['%s' % palavras]=(palavras in doc)
    return caracteristicasTeste




#caracteristicasfrase = extratorpalavras()

basecompletaTreinamento = nltk.classify.apply_features(extratorpalavrasTreinamento,FrasesTreinamentoSemRadical)
basecompletaTeste = nltk.classify.apply_features(extratorpalavrasTeste,FrasestesteSemRadical)


#print(basecompleta)
#CONSTRUIR A TABELA DE PROBABILIDADES



#mostrando as classes

classificador = nltk.NaiveBayesClassifier.train(basecompletaTreinamento) ##constroi as tabelas de probabilidade

acuracia=nltk.classify.accuracy(classificador,basecompletaTeste)
print(nltk.classify.accuracy(classificador,basecompletaTeste)) #acuracia


##############################################TESTE DE PREDIÇÃO########################


def buscapalavrasPredicao(frases):
    todaspalavras=[]
    for (Phrase) in frases:
        todaspalavras.extend(FrasesPredicaoSemRadical)
    return todaspalavras
   






stopwordscompleto = nltk.corpus.stopwords.words('english')

for sentenca in base_comentario_predicao_lista:
    Phrase = sentenca[1]
#    print(Phrase)
    
    radical=nltk.stem.SnowballStemmer('english')
    frasesSemRadicalPredicao = []
#    FrasesPredicaoSemRadical=[str(radical.stem(p)) for p in Phrase.split()]
    FrasesPredicaoSemRadical=[str(radical.stem(p)) for p in Phrase.split()  if p not in stopwordscompleto]
    #print(FrasesPredicaoSemRadical)
    
    palavrasTestePredicao=buscapalavrasPredicao(FrasesPredicaoSemRadical)
    #print(palavrasTestePredicao)
    
    
    FrequenciaPredicao = nltk.FreqDist(palavrasTestePredicao)
    #print(FrequenciaPredicao)
    
    
    palavraunicasPredicao=FrequenciaPredicao.keys()
    #print(palavraunicasPredicao)
    
    
    def extratorpalavrasPredicao(documento):
        doc = set(documento)
        caracteristicasPredicao={}
        for palavras in palavraunicasPredicao:
            caracteristicasPredicao['%s' % palavras]=(palavras in doc)
        return caracteristicasPredicao
    
    basecompletaPredicao =   extratorpalavrasPredicao(palavraunicasPredicao)
    
    print(sentenca[0]+','+classificador.classify(basecompletaPredicao))
    resultado=sentenca[0]+','+classificador.classify(basecompletaPredicao)
#    print(classificador.classify(basecompletaPredicao[0]))
    
    arquivo=open('resultado.txt','a')
    arquivo.write(resultado+'\n')
    arquivo.close()
###############################FIMMMM   DA       PREDICAO##########################################


###################MONTANDO A MATRIZ DE CONFUSAO
acuracia1=nltk.classify.accuracy(classificador,basecompletaTeste)
print(nltk.classify.accuracy(classificador,basecompletaTeste)) #acuracia

esperado=[]
previsto=[]
for (Phrases,Sentiment) in basecompletaTeste:
    resultado=classificador.classify(Phrases)
    previsto.append(resultado)
    esperado.append(Sentiment)


#print(pre//////////////////visto)
#print(esperado)

matriz = ConfusionMatrix(esperado,previsto)
print(matriz)


