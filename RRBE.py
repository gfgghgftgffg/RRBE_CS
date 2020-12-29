from utils import *
import numpy
from PIL import Image
from math import ceil

def RRBE(path, embedRate):
    path = 'c://Users//ASUS//Desktop//origin.jpg'
    embedRate = 0.2
    embedding_threshold_A = 0.25
    embedding_threshold_B = 0.2

    img = Image.open(path)
    img = trans_to_GreyScale(img)
    width, height = img.size
    max_data_length = embedRate * height * width

    if embedRate < embedding_threshold_A:
        multi_embedding_turn = False
        A_height = ceil(max_data_length / width)
    else:
        multi_embedding_turn = True
        A_height = ceil(max_data_length / width / 2)
    A, B, index = divide_img(img, A_height)


RRBE(1,2)