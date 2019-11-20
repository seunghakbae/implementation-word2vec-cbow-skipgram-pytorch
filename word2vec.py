import torch
import math
from collections import Counter
import argparse
import random
from huffman import HuffmanCoding
import time
import numpy as np

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

def NS_Skipgram(centerWord, inputMatrix, outputMatrix):

    V, D = inputMatrix.size()

    inputVector = inputMatrix[centerWord]
    out = outputMatrix.mm(inputVector.view(D, 1))

    loss = 0

    target = out[0]
    NS_samples = out[1:]

    loss = -torch.log(torch.sigmoid(target)) - torch.sum(torch.log(torch.sigmoid(-NS_samples)))

    sigmoid_value = torch.sigmoid(out)
    grad = sigmoid_value

    grad[0][0] -= 1.0

    grad_emb = grad.view(1, -1).mm(outputMatrix)  # /len(contextWords)
    grad_out = grad.mm(inputVector.view(1, -1))

    return loss, grad_emb, grad_out

def HS_Skipgram(centerWord, contextCode, inputMatrix, outputMatrix):
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

    D = inputMatrix.shape[1]
    K = outputMatrix.shape[0]

    input_vector = inputMatrix[centerWord].view(1, D)
    out = torch.mm(input_vector, outputMatrix.t())  # [1 D] X [D K] = [1 K]

    out_mul = 1.0

    for path_node in range(K):
        if contextCode[path_node] == '0':
            out_mul *= torch.sigmoid(out[0][path_node])
        else:
            out_mul *= torch.sigmoid(-out[0][path_node])

    loss = -torch.log(out_mul)

    grad = torch.sigmoid(out)  # [1 K]

    for path_node in range(K):
        if contextCode[path_node] == '0':
            grad[0][path_node] -= 1.0

    grad_out = torch.mm(grad.t(), input_vector)  # [K 1] X [1 D] = [K D]
    grad_emb = torch.mm(grad, outputMatrix)  # [1 K] X [K D] = [1 D]

    return loss, grad_emb, grad_out


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

    V, D = inputMatrix.size()
    inputVector = inputMatrix[centerWord]

    out = outputMatrix.mm(inputVector.view(D, 1))
    expout = torch.exp(out)
    softmax = expout / expout.sum()

    loss = -torch.log(softmax[contextWord])

    grad = softmax
    grad[contextWord] -= 1

    grad_emb = grad.view(1, -1).mm(outputMatrix)
    grad_out = grad.mm(inputVector.view(1, -1))

    return loss, grad_emb, grad_out

def NS_CBOW(contextWords, inputMatrix, outputMatrix):
    V, D = inputMatrix.size()

    inputVector = inputMatrix[contextWords].sum(0)
    out = outputMatrix.mm(inputVector.view(D, 1))

    sigmoid_value = torch.sigmoid(out)
    grad = sigmoid_value

    target = out[0]
    NS_samples = out[1:]

    loss = -torch.log(torch.sigmoid(target)) - torch.sum(torch.log(torch.sigmoid(-NS_samples)))

    grad[0][0] -= 1.0

    grad_emb = grad.view(1, -1).mm(outputMatrix)  # /len(contextWords)
    grad_out = grad.mm(inputVector.view(1, -1))

    return loss, grad_emb, grad_out


def HS_CBOW(centerCode, contextWords, inputMatrix, outputMatrix):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWords : Indices of contextwords (type:list(int))               #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################

###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_emb : Gradient of word embedding (type:torch.tensor(1,D))        #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    D = inputMatrix.shape[1]
    C = len(contextWords)
    K = outputMatrix.shape[0]

    inputVector = inputMatrix[contextWords].sum(0).view(1, D) / C
    out = torch.mm(inputVector, outputMatrix.t())  # [1 D] X [D K] = [1 K]

    out_mul = 1.0

    for path_node in range(K):
        if centerCode[path_node] == '0':
            out_mul *= torch.sigmoid(out[0][path_node])
        else:
            out_mul *= torch.sigmoid(-out[0][path_node])

    loss = -torch.log(out_mul)

    grad = torch.sigmoid(out)  # [1 K]

    for path_node in range(K):
        if centerCode[path_node] == '0':
            grad[0][path_node] -= 1.0

    grad_out = torch.mm(grad.t(), inputVector)  # [K 1] X [1 D] = [K D]
    grad_in = torch.mm(grad, outputMatrix)  # [1 K] X [K D] = [1 D]

    return loss, grad_in, grad_out


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

    V,D = inputMatrix.size()

    inputVector = inputMatrix[contextWords].sum(0)
    out = outputMatrix.mm(inputVector.view(D,1))
    expout = torch.exp(out)
    softmax = expout / expout.sum()

    loss = -torch.log(softmax[centerWord])

    grad = softmax
    grad[centerWord] -= 1

    grad_emb = grad.view(1, -1).mm(outputMatrix)  # /len(contextWords)
    grad_out = grad.mm(inputVector.view(1, -1))

    return loss, grad_emb, grad_out


def word2vec_trainer(corpus, word2ind, codes=None,freqtable = None, nonleaf_ind = None, mode="CBOW", mode2 = "None",use_subsample=None ,dimension=100, learning_rate=0.025, iteration=50000):

# Xavier initialization of weight matrices
    W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    W_out = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    window_size = 5

    losses=[]

    # 각 node에 번호를 붙여준다.
    node = {}
    node[''] = 0
    node_index = 1

    for x in codes:
        for idx in range(1, len(codes[x])):
            if codes[x][:idx] in node:
                pass
            else:
                node[codes[x][:idx]] = node_index
                node_index += 1

    for i in range(iteration):
        #Training word2vec using SGD
        centerword, context = getRandomContext(corpus, window_size)
        centerInd = word2ind[centerword]
        contextInds = [word2ind[word] for word in context]
        centerCode = codes[centerInd]

        if mode=="CBOW":

            if mode2 == "None":
                L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out)
                W_emb[contextInds] -= learning_rate*G_emb
                W_out -= learning_rate*G_out
                losses.append(L.item())

            elif mode2 == "0":
                path = []
                code = centerCode

                for idx in range(len(code)):
                    node_num = node[code[:idx]]
                    path.append(node_num)

                activated = path
                L, G_emb, G_out = HS_CBOW(centerCode, contextInds , W_emb, W_out[activated])
                W_emb[contextInds] -= learning_rate * G_emb
                W_out[activated] -= learning_rate * G_out
                losses.append(L.item())
            else:
                selected = random.sample(freqtable, int(mode2))
                if centerInd in selected:
                    selected.remove(centerInd)
                activated = [centerInd] + selected
                L, G_emb, G_out = NS_CBOW(contextInds, W_emb, W_out[activated])
                W_emb[contextInds] -= learning_rate * G_emb
                W_out[activated] -= learning_rate * G_out
                losses.append(L.item())

        elif mode=="SG":

            if mode2 == "None":
                for contextInd in contextInds:
                    L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out)
                    W_emb[centerInd] -= learning_rate*G_emb.squeeze()
                    W_out -= learning_rate*G_out
                    losses.append(L.item())
            elif mode2 == "0":
                for contextInd in contextInds:

                    path = []
                    contextCode = codes[contextInd]

                    for idx in range(len(contextCode)):
                        node_num = node[contextCode[:idx]]
                        path.append(node_num)

                    activated = path

                    L, G_emb, G_out = HS_Skipgram(centerInd, contextCode, W_emb, W_out[activated])
                    W_emb[centerInd] -= learning_rate * G_emb.squeeze()
                    W_out[activated] -= learning_rate * G_out
                    losses.append(L.item())

            else:
                for contextInd in contextInds:
                    selected = random.sample(freqtable, int(mode2))
                    if contextInd in selected:
                        selected.remove(contextInd)
                    activated = [contextInd] + selected
                    L, G_emb, G_out = NS_Skipgram(centerInd, W_emb, W_out[activated])
                    W_emb[centerInd] -= learning_rate * G_emb.squeeze()
                    W_out[activated] -= learning_rate * G_out
                    losses.append(L.item())

        else:
            print("Unkwnown mode : "+mode)
            exit()

        if i%10000==0:
            avg_loss=sum(losses)/len(losses)
            print("Loss : %f" %(avg_loss,))
            losses=[]

    return W_emb, W_out


def Analogical_Reasoning_Task(embedding, w2i, i2w):
    #######################  Input  #########################
    # embedding : Word embedding (type:torch.tesnor(V,D))   #
    #########################################################
    start_time = time.time()
    embedding = torch.tensor(embedding)
    # print(w2i)
    f = open("questions-words.txt", 'r')
    g = open("result.txt", 'w')
    total_question = 0.0
    total_correct = 0.0
    N = embedding.shape[0]

    vector = {}
    for word in w2i:
        vector[word] = embedding[w2i[word]]

    l2_norm = {}
    for word in w2i:
        l2_norm[word] = torch.dist(vector[word], torch.zeros_like(vector[word]), 2)

    num_questions = 0

    while True:
        line = f.readline()
        if not line: break
        if line[0] == ':': continue
        Qwords = line.split()

        if num_questions % 100 == 0:
            print(str(num_questions) + " read ")
            num_questions+= 1
        else:
            num_questions += 1

        if Qwords[0].lower() in w2i and Qwords[1].lower() in w2i and Qwords[2].lower() in w2i and Qwords[
            3].lower() in w2i:
            total_question += 1.0
            word_1_vec = vector[Qwords[0].lower()]
            word_2_vec = vector[Qwords[1].lower()]
            word_3_vec = vector[Qwords[2].lower()]
            x_vector = word_2_vec - word_1_vec + word_3_vec
            x_vector = torch.tensor(x_vector)
            x_vector_l2_norm = torch.dist(x_vector, torch.zeros_like(x_vector), 2)
            best_similarity = -1.0
            best_idx = None
            for word in w2i:
                word_idx = w2i[word]
                wordvec = vector[word]
                similarity = torch.dot(x_vector, wordvec) / (x_vector_l2_norm * l2_norm[word])
                if similarity > best_similarity and word_idx != w2i[Qwords[0].lower()] and word_idx != w2i[
                    Qwords[1].lower()] and word_idx != w2i[Qwords[2].lower()]:
                    # if similarity > best_similarity # 문제에 나온 단어들도 포함시키고 싶으면 위 조건문을 주석처리하고 이 문장을 사용
                    best_similarity = similarity
                    best_idx = word_idx
            # print(best_similarity)
            if Qwords[3].lower() == i2w[best_idx].lower():
                g.write(
                    "%s %s : %s %s?, Correct !!! \n" % (Qwords[0], Qwords[1], Qwords[2], i2w[best_idx].capitalize()))
                total_correct += 1.0
            else:
                g.write("%s %s : %s %s?, Wrong !!! Answer is %s \n" % (
                Qwords[0], Qwords[1], Qwords[2], i2w[best_idx].capitalize(), Qwords[3]))
    g.write("Questions = %d, Correct Questions = %d, Hitting Rate = %.4f%% \n" % (
    total_question, total_correct, (total_correct / total_question) * 100.0))
    end_time = time.time()
    time_elapsed = end_time - start_time
    g.write("Time Elapsed for Validaiton %02d:%02d:%02d\n" % (
    time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    f.close()
    g.close()
    print('Questions = %d, Correct Questions = %d, Hitting Rate = %.4f' % (
    total_question, total_correct, (total_correct / total_question) * 100.0))
    print('Time Elapsed for Validaiton %02d:%02d:%02d' % (
    time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    parser.add_argument('mode2', metavar='mode2', type=str,
                        help = "0 for Hierarchical Softmax, 1 or more for Negative Sampling, 'None' for None of two")
    parser.add_argument('use_subsampling', metavar='subsample', type=str,
                        help="0 for not using subsampling, 1 for using subsampling")
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    mode2 = args.mode2
    subsample = args.use_subsampling

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

    # Create Huffman Coding
    freq = dict()
    freq[0] = 0

    total_freq = 0

    for word in vocabulary:
        freq[word2ind[word]] = frequency[word]
        total_freq += frequency[word]

    # subsampling
    if subsample == "1":

        freq_subsampling = {}
        for word in vocabulary:
            freq_subsampling[word] = frequency[word] / total_freq

        # calculate subsampling_probability
        prob_subsampling = {}


        for word in vocabulary:
            prob_subsampling[word] = max(0, 1 - math.sqrt(0.001 / freq_subsampling[word]))

        # print(prob_subsampling)
        # exit()

        subsampled_corpus = []
        discard = 0

        for word in processed:
                prob = prob_subsampling[word]
                random_prob = np.random.rand()
                if random_prob > prob:
                    subsampled_corpus.append(word)
                else:
                    discard += 1

        print(len(processed))
        print("Discard : " + str(discard))

        processed = subsampled_corpus


    huffmanCode = HuffmanCoding()
    codes, nonleaf_ind = huffmanCode.build(freq)

    # negative sampling
    freqtable = [0, 0, 0]
    for k, v in frequency.items():
        f = int(v ** 0.75)
        for _ in range(f):
            if k in word2ind.keys():
                freqtable.append(word2ind[k])

    #Training section
    emb,_ = word2vec_trainer(processed, word2ind, codes=codes,freqtable = freqtable ,nonleaf_ind = nonleaf_ind , mode=mode, mode2=mode2, use_subsample=subsample, dimension=64, learning_rate=0.05, iteration=50000)

    Analogical_Reasoning_Task(emb, word2ind, ind2word)


main()