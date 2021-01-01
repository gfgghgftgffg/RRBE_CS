import numpy as np
from PIL import Image
from math import ceil

def replace_lowbit(d, b):
    if d % 2 == b:
        return d
    elif d % 2 == 0:
        return d + 1
    else:
        return d - 1


def embeddingA(path, extra_info=''):
    img = Image.open('c://Users//ASUS//Desktop//enc.png')
    matrix_input = np.asarray(img)
    matrix_input = matrix_input.astype(int)
    height, width = matrix_input.shape

    embedRate = 0.2
    embedding_threshold_A = 0.25
    embedding_threshold_B = 0.2
    max_data_length = embedRate * height * width

    if embedRate < embedding_threshold_A:
        multi_embedding_turn = False
        A_height = ceil(max_data_length / width)
    else:
        multi_embedding_turn = True
        A_height = ceil(max_data_length / width / 2)

    # handle embedding data
    data = '20174345qy'
    dataList = []
    for i in data:
        tmp = bin(ord(i))[2:]
        for j in range(8-len(tmp)):
            dataList.append(0)
        for j in tmp:
            dataList.append(int(j))
    
    dataList += [0 for i in range(20)]
    
    if len(dataList) > max_data_length:
        print('embedding data too big.')
        return img
    

    # embedding
    num = 0
    for i in range(A_height):
        for j in range(width):
            num += 1

            if num <= 19:
                continue
            elif num == 20:
                matrix_input[i,j] = replace_lowbit(matrix_input[i,j], 1)
            elif num > len(dataList) + 20:
                break
            else:
                matrix_input[i,j] = replace_lowbit(matrix_input[i,j], dataList[num-21])
            
        if num > len(dataList) + 20:
            break
    

    #print("done")
    matrix_output = Image.fromarray(matrix_input).convert('L')
    matrix_output.save('c://Users//ASUS//Desktop//enc_s.png')


embeddingA(1,2)
    
