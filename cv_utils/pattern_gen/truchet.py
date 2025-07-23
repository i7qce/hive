import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create Tiles
# See this article for a good discussion:
# https://medium.com/@adbaysal/exploring-truchet-tiles-da61f02981a0

def create_truchet_tiles(N=11):
    i, j = np.indices((N, N))
    tile_1 = (i + j == N//2).astype(float) + (i+j == N - 1 + N//2).astype(float) + (i+j == N - 1 + N//2 + 1).astype(float) + (i + j == N//2 - 1).astype(float)
    tile_2 = np.rot90(tile_1)

    tile_1_c1 = (i + j <= N//2 - 1).astype(float) + (i+j >= N - 1 + N//2 +1).astype(float)
    tile_2_c1 = np.rot90(tile_1_c1)

    tile_1_c2 = 1 - tile_1_c1
    tile_2_c2 = 1 - tile_2_c1

    return tile_1, tile_2, tile_1_c1, tile_2_c1, tile_1_c2, tile_2_c2

def tile_truchet(N_A=21, tile_type=0):
    
    Ai, Aj = np.indices((N_A, N_A))

    if tile_type == 0:
        A = np.random.randint(0,2,(N_A,N_A))
    elif tile_type == 1:
        A = np.bitwise_count(np.bitwise_xor(Ai,Aj)) & 1
    elif tile_type == 2:
        A = np.bitwise_count(Ai + Aj) & 1
    elif tile_type == 3:
        A = np.bitwise_count(Ai * Aj) & 1
    elif tile_type == 4:
        A = np.bitwise_count(Ai & Aj) & 1
    elif tile_type == 5:
        A = np.bitwise_count(Ai | Aj) & 1
    elif tile_type == 6:
        A = np.bitwise_count(Ai ** Aj) & 1
    elif tile_type == 7:
        A = np.bitwise_count((Ai ** Aj) + (Aj ** Ai)) & 1


    A = A + 1
    A = A.astype(np.int32)
    
    return A

def is_tile_valid(num1, num2):
    if num1 == num2:
        return False
    if np.abs(num1) != np.abs(num2) and np.sign(num1) != np.sign(num2):
        return False
    return True

def color(A):
    C = A.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i==0 and j == 0:
                continue
            elif j == 0:
                # we are on first entry of row, check value above
                if not is_tile_valid(C[i-1][j], C[i][j]):
                    C[i][j] = - C[i][j]
            else:
                # we are inside a row, check value to the left
                if not is_tile_valid(C[i][j-1], C[i][j]):
                    C[i][j] = - C[i][j]
    return C

def generate(tile_type, tile_size, num_tiles):
    tile_1, tile_2, tile_1_c1, tile_2_c1, tile_1_c2, tile_2_c2 = create_truchet_tiles(N=tile_size)
    # plt.imshow(tile_2_c1)
    A = tile_truchet(N_A=num_tiles, tile_type=tile_type)
    C = color(A)

    # Outline only
    result1 = np.kron(A==1,  tile_1) + np.kron(A==2, tile_2)

    result = np.kron(C == 1,  tile_1_c1) + np.kron(C == 2, tile_2_c1) +  np.kron(C == -1,  tile_1_c2) + np.kron(C == -2, tile_2_c2)

    return result, result1