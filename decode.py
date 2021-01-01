import utils
import random

def stream_cipher_decrypt(img_matrix, cipher, A_height):
    cipher = utils.md5(cipher)
    random.seed(cipher)
    height, width = img_matrix.shape
    for i in range(0,A_height):
        for j in range(width):
            tmp = int(random.random() * 1000 % 256)
            tmp = tmp-1 if tmp % 2 == 1 else tmp
            img_matrix[i,j] = img_matrix[i,j] ^ tmp
    
    for i in range(A_height,height):
        for j in range(width):
            tmp = int(random.random() * 1000 % 256)
            img_matrix[i,j] = img_matrix[i,j] ^ tmp


    return img_matrix