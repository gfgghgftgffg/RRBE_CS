from encode import *
from PIL import Image
import numpy as np

def trans_to_GreyScale(img):
    if img.mode != 'L':
        img = img.convert('L')
    return img


def divide_img(img, A_height):
    width, height = img.size
    N = height - A_height + 1
    smoothList = [0] * N

    for i in range(N):
        A = img.crop((0, i, width, i+A_height))
        A_width, A_height = A.size

        A = A.load()
        for row in range(A_height):
            for column in range(A_width):
                
                # block A only one row
                if A_height == 1:
                    if row == 0 and column == 0:
                        error = abs(A[column,row] - A[column+1,row])
                    elif row == 0 and column == A_width-1:
                        error = abs(A[column,row] - A[column-1,row])
                    else:
                        error = abs( A[column,row] - (A[column-1,row]+A[column+1,row]) / 2 )

                else:
                    # 4 points
                    if row == 0 and column == 0:
                        error = abs( A[column,row] - (A[column+1,row]+A[column,row+1]) / 2 )
                    elif row == 0 and column == A_width-1:
                        error = abs( A[column,row] - (A[column-1,row]+A[column,row+1]) / 2 )
                    elif row == A_height-1 and column == 0:
                        error = abs( A[column,row] - (A[column+1,row]+A[column,row-1]) / 2 )
                    elif row == A_height-1 and column == A_width-1:
                        error = abs( A[column,row] - (A[column-1,row]+A[column,row-1]) / 2 )
                    # 4 rows/columns
                    elif row == 0:
                        error = abs( A[column,row] - (A[column-1,row]+A[column+1,row]+A[column,row+1]) / 3 )
                    elif row == A_height-1:
                        error = abs( A[column,row] - (A[column-1,row]+A[column+1,row]+A[column,row-1]) / 3 )
                    elif column == 0:
                        error = abs( A[column,row] - (A[column,row-1]+A[column,row+1]+A[column+1,row]) / 3 )
                    elif column == A_width-1:
                        error = abs( A[column,row] - (A[column,row-1]+A[column,row+1]+A[column-1,row]) / 3 )
                    # other inner point
                    else:
                        error = abs( A[column,row] - (A[column,row-1]+A[column,row+1]+A[column-1,row]+A[column+1,row]) / 4 )
                
                smoothList[i] += error
        smoothList[i] = smoothList[i] / (A_height*A_width)
    
    # find least smooth part A, then crop it.
    index = smoothList.index(max(smoothList))
    if index == N-1:
        partA = img.crop((0,index,width,height))
        partB = img.crop((0,0,width,index))
    elif index == 0:
        partA = img.crop((0,0,width,A_height))
        partB = img.crop((0,A_height,width,height))
    else:
        partA = img.crop((0,index,width,index+A_height))
        partB1 = img.crop((0,0,width,index))
        partB2 = img.crop((0,index+A_height,width,height))
        partB = Image.new('L', (width,height-A_height))
        partB.paste(partB1, (0,0,width,index))
        partB.paste(partB2,(0,index,width,height-A_height))
    
    return partA,partB,index


def divide_img2(img, A_height):
    matrix_input = np.asarray(img)
    matrix_input = matrix_input.astype(int)
    height, width = matrix_input.shape
    N = height - A_height + 1
    smoothList = [0] * N

    for i in range(N):
        A = matrix_input[i:i+A_height,:]
        #A = img.crop((0, i, width, i+A_height))
        A_height, A_width = A.shape

        for row in range(A_height):
            for column in range(A_width):
                
                # block A only one row
                if A_height == 1:
                    if row == 0 and column == 0:
                        error = abs(A[row,column] - A[row,column+1])
                    elif row == 0 and column == A_width-1:
                        error = abs(A[row,column] - A[row,column-1])
                    else:
                        error = abs( A[row,column] - (A[row,column-1]+A[row,column+1]) / 2 )

                else:
                    # 4 points
                    if row == 0 and column == 0:
                        error = abs( A[row,column] - (A[row,column+1]+A[row+1,column]) / 2 )
                    elif row == 0 and column == A_width-1:
                        error = abs( A[row,column] - (A[row,column-1]+A[row+1,column]) / 2 )
                    elif row == A_height-1 and column == 0:
                        error = abs( A[row,column] - (A[row,column+1]+A[row-1,column]) / 2 )
                    elif row == A_height-1 and column == A_width-1:
                        error = abs( A[row,column] - (A[row,column-1]+A[row-1,column]) / 2 )
                    # 4 rows/columns
                    elif row == 0:
                        error = abs( A[row,column] - (A[row,column-1]+A[row,column+1]+A[row+1,column]) / 3 )
                    elif row == A_height-1:
                        error = abs( A[row,column] - (A[row,column-1]+A[row,column+1]+A[row-1,column]) / 3 )
                    elif column == 0:
                        error = abs( A[row,column] - (A[row-1,column]+A[row+1,column]+A[row,column+1]) / 3 )
                    elif column == A_width-1:
                        error = abs( A[row,column] - (A[row-1,column]+A[row+1,column]+A[row,column-1]) / 3 )
                    # other inner point
                    else:
                        error = abs( A[row,column] - (A[row-1,column]+A[row+1,column]+A[row,column-1]+A[row,column+1]) / 4 )
                
                smoothList[i] += error
        smoothList[i] = smoothList[i] / (A_height*A_width)
    
    # find least smooth part A, then crop it.
    index = smoothList.index(max(smoothList))
    if index == N-1:
        partA = matrix_input[index:height,:]
        partB = matrix_input[0:index,:]
        #partA = img.crop((0,index,width,height))
        #partB = img.crop((0,0,width,index))
    elif index == 0:
        partA = matrix_input[0:A_height,:]
        partB = matrix_input[A_height:height,:]
        #partA = img.crop((0,0,width,A_height))
        #partB = img.crop((0,A_height,width,height))
    else:
        partA = matrix_input[index:index+A_height,:]
        partB1 = matrix_input[0:index,:]
        partB2 = matrix_input[index+A_height:height,:]
        partB = np.concatenate((partB1, partB2))
        # partA = img.crop((0,index,width,index+A_height))
        # partB1 = img.crop((0,0,width,index))
        # partB2 = img.crop((0,index+A_height,width,height))
        # partB = Image.new('L', (width,height-A_height))
        # partB.paste(partB1, (0,0,width,index))
        # partB.paste(partB2,(0,index,width,height-A_height))
    
    partA = Image.fromarray(partA).convert('L')
    partB = Image.fromarray(partB).convert('L')
    
    return partA,partB,index




def embedB(imgB, dataList, embedding_threshold_B):
    matrix_input = np.asarray(imgB)
    matrix_input = matrix_input.astype(int)
    matrix_output = encode(matrix_input, dataList)
    #matrix_output = encode(imgB, dataList)
    return matrix_output


def encrypt(img_matrix, cipher):
    img_matrix_copy = img_matrix.copy()
    encrypted_img = stream_cipher_encrypt(img_matrix_copy, cipher)
    flagData = [1 for i in range(10)]
    flagData += [0,0,1] # 001_bin = 1_oc
    flagData += [1,0] # [encrypt_flag, embeddinmg flag]

    if encrypted_img.shape[0]*encrypted_img.shape[1] < 15:
        print("not enough space to embedding.")
        return img_matrix

    num = 0
    for i in range(encrypted_img.shape[0]):
        for j in range(encrypted_img.shape[1]):
            encrypted_img[i,j] = flagData[num]
            num += 1
            if num > 14:
                break
        if num > 14:
            break
        
    return encrypted_img