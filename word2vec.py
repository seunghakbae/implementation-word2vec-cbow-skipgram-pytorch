import torch
from random import shuffle
import random
from collections import Counter
import argparse


def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)

    context = corpus[max(0, wordID - C):wordID]
    if wordID+1 < len(corpus):
        context += corpus[wordID+1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]

    context = [w for w in context if w != centerword]


    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)


def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWord : Index of a contextword (type:int)                       #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################

###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word vector (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

###############################  forward path  ################################

    x = torch.zeros(1, (len(inputMatrix))) # make center-word one-hot vector
    x[0][centerWord] = 1
    v = torch.mm(x,inputMatrix) # x dot product inputMatrix
    z = torch.mm(outputMatrix, v.t()) # v dot product outputMatrix

    #softmax
    e = torch.exp(z)
    softmax = e /torch.sum(e, dim=0, keepdim=True)

    print(contextWord)
    print(softmax[contextWord][0])

    # loss
    negative = -torch.log(softmax) # softmax 적용
    loss = negative[contextWord][0] # 정답 label만 출력.

###############################################################################

###############################  backward path  ################################
    softmax_clone = torch.clone(softmax)
    softmax_clone[contextWord][0] = softmax[contextWord][0] - 1 # dL/dy * dy/dz
    e_loss = softmax_clone # e_loss = dL/dy * dy/dz
    dv = torch.mm(e_loss.t(), outputMatrix) # dL/dz = (dL/dy * dy/dz) * dz/dv
    doutputMatrix = torch.mm(e_loss, v) # dL/doutputMatrix = (dL/dy * dy/dz) * dz/doutputMatrix

    grad_out = doutputMatrix # gradient of outputMatrix
    grad_emb = dv # gradient of inputMatrix
###############################################################################

    return loss, grad_emb, grad_out

def CBOW(centerWord, contextWords, inputMatrix, outputMatrix):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWords : Indices of contextwords (type:list(int))               #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################

###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word embedding (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

###############################  forward path  ################################

    v_sum = torch.zeros((1,64)) # sum of v
    x_sum = torch.zeros((1, len(inputMatrix))) # sum of x

    for index in contextWords: # index of contextWords
        x = torch.zeros(1, (len(inputMatrix))) # each x
        x[0][index] = 1 # one-hot vector
        x_sum += x
        v = torch.mm(x,inputMatrix) # x dot product inputMatrix
        v_sum += v # add all v

    # print(v_sum)
    v_sum = v_sum / len(contextWords) # divide by num of x
    z = torch.mm(outputMatrix, v_sum.t()) # v_sum dot product outputMatrix

    #softmax
    e = torch.exp(z)
    softmax = e /torch.sum(e, dim=0, keepdim=True)
    # print(softmax)

    # loss
    negative = -torch.log(softmax) # softmax 적용
    loss = negative[centerWord][0] # 정답 label만 출력.
###############################################################################

###############################  backward path  ################################
    softmax_clone = torch.clone(softmax)
    softmax_clone[centerWord][0] = softmax[centerWord][0] - 1 # dL/dy * dy/dz
    e_loss = softmax_clone # e_loss = dL/dy * dy/dz
    dv_sum = torch.mm(e_loss.t(), outputMatrix) # dL/dv_sum = (dL/dy * dy/dz) * dz/dv_sum
    doutputMatrix = torch.mm(e_loss, v_sum) # dL/doutputMatrix = (dL/dy * dy/dz) * dz/doutputMatrix
    # dinputMatrix = torch.mm((x_sum / len(contextWords)).t(), dv_sum) # dL/dinputMatrix = (dL/dy * dy/dz) * dz/dv_sum * dv_sum/dinputMatrix

    grad_out = doutputMatrix # gradient of outputMatrix
    grad_emb = dv_sum # gradient of inputMatrix
###############################################################################

    return loss, grad_emb, grad_out


def word2vec_trainer(corpus, word2ind, mode="CBOW", dimension=100, learning_rate=0.025, iteration=50000):

# Xavier initialization of weight matrices
    W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    W_out = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    window_size = 5


    losses=[]
    for i in range(iteration):
        #Training word2vec using SGD
        centerword, context = getRandomContext(corpus, window_size)
        centerInd = word2ind[centerword]
        contextInds =[word2ind[word] for word in context]

        if mode=="CBOW":
            L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out)
            W_emb[contextInds] -= learning_rate*G_emb
            W_out -= learning_rate*G_out
            losses.append(L.item())

        elif mode=="SG":
            for contextInd in contextInds:
                L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out)
                W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                W_out -= learning_rate*G_emb
                losses.append(L.item())

        else:
            print("Unkwnown mode : "+mode)
            exit()

        if i%10000==0:
        	avg_loss=sum(losses)/len(losses)
        	print("Loss : %f" %(avg_loss,))
        	losses=[]

    return W_emb, W_out


def sim(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2ind[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    part = args.part

	#Load and tokenize corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("tokenizing...")
    corpus = text.split()
    frequency = Counter(corpus)
    processed = []
    #Discard rare words
    for word in corpus:
        if frequency[word]>4:
            processed.append(word)

    vocabulary = set(processed)

    #Assign an index number to a word
    word2ind = {}
    word2ind[" "]=0
    i = 1
    for word in vocabulary:
        word2ind[word] = i
        i+=1
    ind2word = {}
    for k,v in word2ind.items():
        ind2word[v]=k

    print("Vocabulary size")
    print(len(word2ind))
    print()

    #Training section
    emb,_ = word2vec_trainer(processed, word2ind, mode=mode, dimension=64, learning_rate=0.05, iteration=50000)
    
    #Print similar words
    testwords = ["one", "are", "he", "have", "many", "first", "all", "world", "people", "after"]
    for tw in testwords:
    	sim(tw,word2ind,ind2word,emb)

main()