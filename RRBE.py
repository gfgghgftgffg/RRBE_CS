from RRBE_op import *
import numpy
from PIL import Image
from math import ceil,floor

def RRBE(path, embedRate):
    path = 'c://Users//ASUS//Desktop//origin.jpg'
    embedRate = 0.2
    embedding_threshold_A = 0.25
    embedding_threshold_B = 0.2

    img = Image.open(path)
    img = trans_to_GreyScale(img)
    width, height = img.size
    max_data_length = embedRate * height * width

    #Get part A and B.
    if embedRate < embedding_threshold_A:
        multi_embedding_turn = False
        A_height = ceil(max_data_length / width)
    else:
        multi_embedding_turn = True
        A_height = ceil(max_data_length / width / 2)
    A, B, index = divide_img(img, A_height)

    print("Split done.")

    # Extract LSB info from A.
    '''
    dataList = []
    dataA = A.load()
    for i in range(A.size[1]):
        for j in range(A.size[0]):
            dataList.append(dataA[j,i] % 2)
    if multi_embedding_turn:
        for i in range(A.size[1]):
            for j in range(A.size[0]):
                dataList.append( floor(dataA[j,i]/2) % 2)
    '''
    dataList = np.asarray(A)
    dataList = dataList.astype(int)
    matrixA = dataList.copy()
    dataList %= 2
    dataList = list(dataList.flat)
    #print(dataList)
    
    
    # Embedding LSB info from A to B.
    embeddingRound = 1
    print("Embedding round:", embeddingRound)
    handled_matrixB = embedB(B, dataList, embedding_threshold_B)

    print("Embedding done.")

    # concat & encrypt
    handled_matrixB_copy = handled_matrixB.copy()
    handled_img = np.concatenate((matrixA, handled_matrixB))
    cipher = 'test'
    encrypted_img = encrypt(handled_img, cipher)

    image_output = Image.fromarray(encrypted_img).convert('L')
    
    image_output.save('c://Users//ASUS//Desktop//enc.png')

RRBE(1,2)