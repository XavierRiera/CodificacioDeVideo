# Imports
import subprocess
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pywt

###############################################################

# EX 2

# From: https://web.archive.org/web/20180423091842/http://www.equasys.de/colorconversion.html
# Convert from RGB to YUV
def rgb_to_yuv(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b
    return y, u, v

# Convert from YUV to RGB
def yuv_to_rgb(Y, U, V):
    r = Y + 1.140 * V
    g = Y - 0.395 * U - 0.581 * V
    b = Y + 2.032 * U
    return r, g, b

###############################################################

# EX 3

# From: https://trac.ffmpeg.org/wiki/Scaling --> ffmpeg -i input.jpg -vf "scale=iw*2:ih" input_double_width.png
# To run commands using a method https://www.geeksforgeeks.org/python/executing-shell-commands-with-python/

def resize(input_path, iw, ih, output_path):                                    # iw --> width, ih --> height ----> així podem canviar la mida de l'input a la mida que volguem
    comando = f"ffmpeg -i {input_path} -vf scale={iw}:{ih} {output_path}"

    subprocess.run(comando)

###############################################################

# EX 4

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def zigzag(img):                                                                            ########## chat
    h, w = img.shape[:2]
    zigzag_list = []

    for s in range(h + w - 1):
        if s % 2 == 0:
            # even: go bottom→top
            for y in range(min(s, h-1), max(-1, s-w), -1):
                x = s - y
                zigzag_list.append(img[y, x])
        else:
            # odd: go top→bottom
            for y in range(max(0, s-w+1), min(s+1, h)):
                x = s - y
                zigzag_list.append(img[y, x])

    return zigzag_list


def serpentine(input_path):

    image_data = mpimg.imread(input_path)
    h, w = image_data.size
    
    serpentine_data = []
    for x in range(w):
        for y in range(h):
            serpentine_data.append(image_data[x, y])

###############################################################

# EX 5

# Black and white from: https://stackoverflow.com/questions/32384057/ffmpeg-black-and-white-conversion
# Max compression from: https://stackoverflow.com/questions/10225403/how-can-i-extract-a-good-quality-jpeg-image-from-a-video-file-with-ffmpeg/10234065#10234065

def black_and_white_max_compression(input_path, output_path):
    comando = f"ffmpeg -i {input_path} -vf format=gray -q:v 31 {output_path}"
    subprocess.run(comando)


# RLE from: https://www.geeksforgeeks.org/dsa/run-length-encoding/
def RLE(st):
    n = len(st)
    i = 0
    while i < n:
        # Count occurrences of current character
        count = 1
        while i < n - 1 and st[i] == st[i + 1]:
            count += 1
            i += 1
        # Print character and its count
        print(st[i] + str(count), end="")
        i += 1

###############################################################

# EX 6
# From: https://www.tutorialspoint.com/scipy/scipy_dct_function.htm --> norm = 'ortho'
# From: https://stackoverflow.com/questions/13904851/use-pythons-scipy-dct-ii-to-do-2d-or-nd-dct?utm_source=chatgpt.com --> dct dynamic for more than 1D array
# From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idct.html --> idct

import numpy as np
from scipy.fftpack import dct, idct
class DCT:
    def encode (array):
        array = np.array(array, dtype=float)
        for line in range (array.ndim):
            # We use the norm = "ortho" to ensure uniform energy distribution
            result = dct(array, axis=line, norm = "ortho")
        return result

    
    def decode (array):
        array = np.array(array, dtype=float)
        for line in reversed(range(array.ndim)):
            result = idct(array, axis=line, norm = "ortho")
        return result

###############################################################

# EX 7
# From: https://pywavelets.readthedocs.io/en/latest/

def wavelet_transform_example(input_path):
    # Load image
    original = mpimg.imread(input_path)
    print(original)

    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
    
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    
    #r, g, b = 255, 0, 0                                                                     # Only RED in RGB
    #y, u, v = rgb_to_yuv(r, g, b)                                                           # Convert RGB to YUV
    #print(f"RGB({r}, {g}, {b}) to YUV({y:.2f}, {u:.2f}, {v:.2f})")

    #r2, g2, b2 = yuv_to_rgb(y, u, v)                                                        # Convert back YUV to RGB
    #print(f"YUV({y:.2f}, {u:.2f}, {v:.2f}) back to RGB({r2:.2f}, {g2:.2f}, {b2:.2f})")      # Verify the conversion --> no és exactament es mateixos valors RGB per culpa d'arrodoniments

    #resize("ex1.jpg", 100, 100, "output.jpg")
    #black_and_white_max_compression("ex1.jpg", "output_bw.jpg")


    #st = "00011000110011"
    #RLE(st)

    data = [1, 2, 3, 4]
    print(DCT.encode(data))

    #wavelet_transform_example("ex1.jpg")
