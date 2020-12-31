import utils
import random


# 得到B边缘像素LSB
def get_margin_pixel(matrix):
    height, width = matrix.shape
    margin_pixels = matrix[0].tolist() + matrix[-1].tolist() + matrix[:, 0][1:height - 1].tolist() + matrix[:, -1][1:height - 1].tolist()
    margin_bits = [utils.get_lowestBit(pixel) for pixel in margin_pixels]
    return margin_bits


def findKey(error, condition):
    height, width = error.shape
    histogram = [0 for i in range(511)]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if condition(i, j):
                #偏置
                histogram[error[i][j] + 255] += 1
    p = q = 0
    for item in histogram:
        if item > histogram[p]:
            q = p
            p = histogram.index(item)
        elif item > histogram[q]:
            q = histogram.index(item)
    lm = p - 255
    rm = q - 255
    if lm > rm:
        lm, rm = rm, lm       
    t = p
    for i in range(0, p + 1):
        if histogram[i] < histogram[t]:
            t = i
    ln = t - 255
    t = q
    for i in range(q, len(histogram)):
        if histogram[i] < histogram[t]:
            t = i
    rn = t - 255
    return [lm, rm, ln, rn]


def additive_expansion(lm, ln, rm, rn, b, e):
    flag = 0
    if e <= lm:
        sine = -1
    elif e >= rm:
        sine = 1
    if e == lm or e == rm:
        e += sine * b
        flag = 1
    elif ln < e < lm or rm < e < rn:
        e += sine
    return e, flag



def encode(matrixB, infoList):
    sample_lm = sample_rm = sample_ln = sample_rn = 0
    dataList = get_margin_pixel(matrixB) + infoList
    dataSize = len(dataList)
    p = 0
    height, width = matrixB.shape
    boundary_map = []

    interpolation_matrixB = utils.generate_interpolation_image(matrixB, 45, utils.is_non_sample_pixel_first)
    interpolation_matrixB = utils.generate_interpolation_image(interpolation_matrixB, 0, utils.is_non_sample_pixel_second)
    error = matrixB - interpolation_matrixB

    notSample_lm, notSample_rm, notSample_ln, notSample_rn = findKey(error, utils.is_non_sample_pixel)

    for i in range(1,  height - 1):
        for j in range(1, width - 1):
            # 将信息插入非采样像素
            if p < dataSize and utils.is_non_sample_pixel(i, j):
                if matrixB[i][j] == 0 or matrixB[i][j] == 255:
                    boundary_map.append(0)
                else:
                    error[i][j], flag = additive_expansion(notSample_lm, notSample_ln, notSample_rm, notSample_rn, dataList[p], error[i][j])
                    p += flag
                    if interpolation_matrixB[i][j] + error[i][j] == 0 or interpolation_matrixB[i][j] + error[i][j] == 255:
                        boundary_map.append(1)

    interpolation_matrixB += error
    l = len(boundary_map)
    # p为已插入的data bit数
    np_size = p


    if p < dataSize:
        # 使用sample pixel插值
        interpolation_matrixB_withSample = utils.generate_interpolation_image(interpolation_matrixB, 0, utils.is_sample_pixel)
        error = interpolation_matrixB - interpolation_matrixB_withSample
        sample_lm, sample_rm, sample_ln, sample_rn = findKey(error, utils.is_sample_pixel)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if p < dataSize and utils.is_sample_pixel(i, j):
                    # ?
                    if interpolation_matrixB[i][j] == 0 or interpolation_matrixB[i][j] == 255:
                        boundary_map.append(0)
                    else:
                        e[i][j], flag = additive_expansion(sample_lm, sample_ln, sample_rm, sample_rn, dataList[p], error[i][j])
                        p += flag

                        if interpolation_matrixB_withSample[i][j] + error[i][j] == 0 or interpolation_matrixB_withSample[i][j] + error[i][j] == 255:
                            boundary_map.append(1)
        interpolation_matrixB_withSample += error
        interpolation_matrixB = interpolation_matrixB_withSample

    sp_size = p - np_size
    # 容量不足
    if p < dataSize:
        print('容量不足')
        return interpolation_matrixB_withSample
    

    #一轮插值完毕
    # print('encode')
    # print('ns_lm, ns_ln, ns_rm, ns_rn: {0}, {1}, {2}, {3}'.format(notSample_lm, notSample_ln, notSample_rm, notSample_rn))
    # print('s_lm, s_ln, s_rm, s_rn: {0}, {1}, {2}, {3}'.format(sample_lm, sample_ln, sample_rm, sample_rn))
    # print('np_size, sp_size: {0}, {1}'.format(np_size, sp_size))
    # print('length: {}'.format(l))
    # print('BoundaryMap: {}', boundary_map)
    # print('dataList: {}'.format(dataList))
    

    sample_lm_bits = utils.int2bits9(sample_lm)
    sample_ln_bits = utils.int2bits9(sample_ln)
    sample_rm_bits = utils.int2bits9(sample_rm)
    sample_rn_bits = utils.int2bits9(sample_rn)
    notSample_lm_bits = utils.int2bits9(notSample_lm)
    notSample_ln_bits = utils.int2bits9(notSample_ln)
    notSample_rm_bits = utils.int2bits9(notSample_rm)
    notSample_rn_bits = utils.int2bits9(notSample_rn)
    l_bits = utils.int2bits_u32(l)
    np_size_bits = utils.int2bits_u32(np_size)
    sp_size_bits = utils.int2bits_u32(sp_size)
    overhead = sample_lm_bits + sample_ln_bits + sample_rm_bits + sample_rn_bits + notSample_lm_bits + notSample_ln_bits + notSample_rm_bits + notSample_rn_bits
    overhead = overhead + l_bits + sp_size_bits + np_size_bits + boundary_map

    # 边界像素不足
    if len(overhead) > (width + height) * 2 - 4:
        print('边界像素不足')
        return interpolation_matrixB
    i = 1
    while i <= len(overhead):
        if i <= width:
            interpolation_matrixB[0][i - 1] = utils.replace_lowbit(interpolation_matrixB[0][i - 1], overhead[i - 1])
        elif i <= 2 * width:
            interpolation_matrixB[height - 1][i - width - 1] = utils.replace_lowbit(interpolation_matrixB[height - 1][i - width - 1], overhead[i - 1])
        elif i <= 2 * width + height - 2:
            interpolation_matrixB[i - 2 * width][0] = utils.replace_lowbit(interpolation_matrixB[i - 2 * width][0], overhead[i - 1])
        elif i <= 2 * width + 2 * height - 4:
            interpolation_matrixB[i - 2 * width - height + 2][width - 1] = utils.replace_lowbit(interpolation_matrixB[i - 2 * width - height + 2][width - 1], overhead[i - 1])
        else:
            print('边缘地区容量不足')
        i += 1
    return interpolation_matrixB


def stream_cipher_encrypt(img_matrix, cipher):
    cipher = utils.md5(cipher)
    random.seed(cipher)
    height, width = img_matrix.shape
    for i in range(height):
        for j in range(width):
            tmp = int(random.random() * 1000 % 256)
            img_matrix[i,j] = img_matrix[i,j] ^ tmp
    return img_matrix
