import encode,decode
from PIL import Image
import numpy as np
import utils

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




def embedB(imgB, dataList, Aindex):
    matrix_input = np.asarray(imgB)
    matrix_input = matrix_input.astype(int)
    matrix_output = encode.encode(matrix_input, dataList, Aindex)
    #matrix_output = encode(imgB, dataList)
    return matrix_output


def encrypt(img_matrix, cipher):
    img_matrix_copy = img_matrix.copy()
    encrypted_img = encode.stream_cipher_encrypt(img_matrix_copy, cipher)
    flagData = [1 for i in range(15)]
    flagData += [0,0,1] # 001_bin = 1_oc
    flagData += [1,0] # [encrypt_flag, embeddinmg flag]

    if encrypted_img.shape[0]*encrypted_img.shape[1] < 20:
        print("not enough space to embedding.")
        return img_matrix

    num = 0
    for i in range(encrypted_img.shape[0]):
        for j in range(encrypted_img.shape[1]):
            encrypted_img[i,j] = utils.replace_lowbit(encrypted_img[i,j], flagData[num])
            num += 1
            if num > 19:
                break
        if num > 19:
            break

    return encrypted_img


def decrypt(img_matrix, cipher, A_height):
    img_matrix_copy = img_matrix.copy()
    decrypted_img = decode.stream_cipher_decrypt(img_matrix_copy, cipher, A_height)

    partA = img_matrix_copy[0:A_height,:]
    partB = img_matrix_copy[A_height:,:]

    margin_info = []
    margin_info += [x for x in (partB[0,:] % 2)]
    margin_info += [x for x in (partB[-1,:] % 2)]
    margin_info += [x for x in (partB[1:-2,0] % 2)]
    margin_info += [x for x in (partB[1:-2,-1] % 2)]

    Aindex = margin_info[0:32]
    Aindex = utils.bits2int_u32(Aindex)
    
    partB1 = partB[0:Aindex,:]
    partB2 = partB[Aindex:,:]

    decrypted_img = np.concatenate((partB1,partA))
    decrypted_img = np.concatenate((decrypted_img,partB2))

    return decrypted_img



def recover(e, lm, ln, rm, rn, payload):
    t = 0
    if e == lm or e == rm:
        payload.append(0)
        t = 1
    elif e == lm - 1 or e == rm + 1:
        payload.append(1)
        t = 1
    if e == lm or e == lm - 1:
        e += payload[-1]
    elif e == rm or e == rm + 1:
        e -= payload[-1]
    elif ln <= e < lm - 1:
        e += 1
    elif rm + 1 < e <= rn:
        e -= 1
    return e, t


def recoveryB(matrixB):
    height, width = matrixB.shape
    margin_info = []
    margin_info += [x for x in (matrixB[0,:] % 2)]
    margin_info += [x for x in (matrixB[-1,:] % 2)]
    margin_info += [x for x in (matrixB[1:-2,0] % 2)]
    margin_info += [x for x in (matrixB[1:-2,-1] % 2)]

    Aindex = utils.bits2int_u32(margin_info[0:32])
    sample_lm = utils.bits2int9(margin_info[32:41])
    sample_ln = utils.bits2int9(margin_info[41:50])
    sample_rm = utils.bits2int9(margin_info[50:59])
    sample_rn = utils.bits2int9(margin_info[59:68])
    notSample_lm = utils.bits2int9(margin_info[68:77])
    notSample_ln = utils.bits2int9(margin_info[77:86])
    notSample_rm = utils.bits2int9(margin_info[86:95])
    notSample_rn = utils.bits2int9(margin_info[95:104])
    l = utils.bits2int_u32(margin_info[104:136])
    sp_size = utils.bits2int_u32(margin_info[136:168])
    np_size = utils.bits2int_u32(margin_info[168:200])
    boundary_map = margin_info[200:]

    s_payload = []
    ns_payload = []

    # info in sample pixel
    if sp_size != 0:
        b_index = l # This is small L, not one
        p = 0
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if utils.is_sample_pixel(i, j) and p < sp_size:
                    inter_pixel = utils.interpolation_pixel(matrixB, i, j, 0)
                    e = matrixB[i][j] - inter_pixel    #e'
                    if matrixB[i][j] > 0 and matrixB[i][j] < 255:
                        e, tmp = recover(e, sample_lm, sample_ln, sample_rm, sample_rn, s_payload)    # e
                        matrixB[i][j] = inter_pixel + e    # x
                        p += tmp
                    else:
                        if boundary_map[b_index] != 0:
                            e, tmp = recover(e, sample_lm, sample_ln, sample_rm, sample_rn, s_payload)
                            matrixB[i][j] = inter_pixel + e
                            p += tmp
                        b_index += 1

    # not sample pixel
    p = 0
    b_index = 0
    # inter_y中含有原始采样像素，和第一类非采样像素的插值值
    inter_B = utils.generate_interpolation_image(matrixB, 45, utils.is_non_sample_pixel_first)
    inter_B = utils.generate_interpolation_image(inter_B, 0, utils.is_non_sample_pixel_second)    # x'

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (p < np_size or sp_size != 0) and utils.is_non_sample_pixel(i, j):
                inter_pixel = inter_B[i][j]
                e = matrixB[i][j] - inter_pixel    # e'
                if 0 < matrixB[i][j] < 255:
                    e, tmp = recover(e, notSample_lm, notSample_ln, notSample_rm, notSample_rn, ns_payload)    # e
                    matrixB[i][j] = inter_pixel + e    # x
                    p += tmp
                else:
                    if boundary_map[b_index] != 0:
                        e, tmp = recover(e, notSample_lm, notSample_ln, notSample_rm, notSample_rn, ns_payload)
                        matrixB[i][j] = inter_pixel + e    # x
                        p += tmp
                    b_index += 1

    dataList = ns_payload + s_payload
    margin_len = (width + height) * 2 - 4
    margin_bits = dataList[0:margin_len]
    Ainfo = dataList[margin_len:]

    # recovery margin B
    k = 0
    for i in range(width):
        matrixB[0][i] = utils.replace_lowbit(matrixB[0][i], margin_bits[k])
        k += 1
    for i in range(width):
        matrixB[height-1][i] = utils.replace_lowbit(matrixB[height-1][i], margin_bits[k])
        k += 1
    for i in range(1, height-1):
        matrixB[i][0] = utils.replace_lowbit(matrixB[i][0], margin_bits[k])
        k += 1
    for i in range(1, height-1):
        matrixB[i][width-1] = utils.replace_lowbit(matrixB[i][width - 1], margin_bits[k])
        k += 1

    return matrixB,Ainfo


def recoveryA(matrixA, Ainfo):
    height, width = matrixA.shape

    k = 0
    for i in range(height):
        for j in range(width):
            if k < len(Ainfo):
                matrixA[i,j] = utils.replace_lowbit(matrixA[i,j], Ainfo[k])
                k += 1
    return matrixA

    


def recoveryAB(img_matrix, A_height):
    height, width = img_matrix.shape
    img_matrix_list = list(img_matrix.flat)
    startFlag = [1 for i in range(15)]
    A_pos = -1

    for i in range(0,len(img_matrix_list),width):
        tmpFlag = img_matrix_list[i:i+15]
        tmpFlag = [x%2 for x in tmpFlag]
        if tmpFlag == startFlag:
            A_pos = i // width
            break
    
    if A_pos < 0:
        print("Not find A")
        return img_matrix
    
    partA = img_matrix[A_pos:A_pos+A_height,:]
    partB = np.concatenate((img_matrix[0:A_pos,:],img_matrix[A_pos+A_height:,:]))

    partB, Ainfo = recoveryB(partB)
    partA = recoveryA(partA, Ainfo)

    partB1 = partB[0:A_pos,:]
    partB2 = partB[A_pos:,:]

    matrix_recoveryed = np.concatenate((partB1,partA))
    matrix_recoveryed = np.concatenate((matrix_recoveryed,partB2))
    return matrix_recoveryed