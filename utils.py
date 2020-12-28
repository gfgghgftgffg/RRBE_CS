from PIL import Image

def trans_to_GreyScale(img):
    if img.mode != 'L':
        img = img.convert('L')
    return img


def divide_img(img, A_height):
    width, height = img.size
    N = height - A_height + 1
    smoothList = [0] * N

    for i in range(N):
        A = img.crop((0, i, width, i-1+A_height))
        A_width, A_height = A.size

        A_data = A.load()
        for row in range(A_height):
            for column in range(A_width):
                
                # block A only one row
                if A_height == 1:
                    if row == 0 and column == 0:
                        error = abs(A_data[row,column] - A_data[row,column+1])
                    elif row == 0 and column == A_width-1:
                        error = abs(A_data[row,column] - A_data[row,column-1])
                    else:
                        error = abs( A_data[row,column] - (A_data[row,column-1]+A_data[row,column-1]) / 2 )

                else:
                    # 4 points
                    if row == 0 and column == 0:
                        error = abs( A_data[row,column] - (A_data[row,column+1]+A_data[row+1,column]) / 2 )
                    elif row == 0 and column == A_width-1:
                        error = abs( A_data[row,column] - (A_data[row,column-1]+A_data[row+1,column]) / 2 )
                    elif row == A_height-1 and column == 0:
                        error = abs( A_data[row,column] - (A_data[row,column+1]+A_data[row-1,column]) / 2 )
                    elif row == A_height-1 and column == A_width-1:
                        error = abs( A_data[row,column] - (A_data[row,column-1]+A_data[row-1,column]) / 2 )
                    # 4 rows/columns
                    elif row == 0:
                        error = abs( A_data[row,column] - (A_data[row,column-1]+A_data[row,column+1]+A_data[row+1,column]) / 3 )
                    elif row == A_height-1:
                        error = abs( A_data[row,column] - (A_data[row,column-1]+A_data[row,column+1]+A_data[row-1,column]) / 3 )
                    elif column == 0:
                        error = abs( A_data[row,column] - (A_data[row-1,column]+A_data[row+1,column]+A_data[row,column+1]) / 3 )
                    elif column == A_width-1:
                        error = abs( A_data[row,column] - (A_data[row-1,column]+A_data[row+1,column]+A_data[row,column-1]) / 3 )
                    # other inner point
                    else:
                        error = abs( A_data[row,column] - (A_data[row-1,column]+A_data[row+1,column]+A_data[row,column-1]+A_data[row,column+1]) / 4 )
                
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

