# From: https://fastapi.tiangolo.com/tutorial/first-steps/#deploy-your-app-optional

from fastapi import FastAPI, UploadFile, File, Response
from pydantic import BaseModel
import subprocess
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from typing import List, Tuple
import base64
from io import BytesIO
import tempfile
import os
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

app = FastAPI()

###############################################################

# EX 2 - RGB to YUV conversion

def rgb_to_yuv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert RGB to YUV color space"""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b
    return y, u, v

def yuv_to_rgb(Y: float, U: float, V: float) -> Tuple[float, float, float]:
    """Convert YUV back to RGB color space"""
    r = Y + 1.140 * V
    g = Y - 0.395 * U - 0.581 * V
    b = Y + 2.032 * U
    return r, g, b

class ColorConversionRequest(BaseModel):
    r: float
    g: float
    b: float

class YUVConversionRequest(BaseModel):
    Y: float
    U: float
    V: float

###############################################################

# EX 3 - Resize images with ffmpeg

def resize(input_path: str, iw: int, ih: int, output_path: str):
    """Resize image using ffmpeg"""
    # Use docker to run ffmpeg in a separate container via a named volume
    vol = os.environ.get("SHARED_VOLUME", "practice1_shared")
    # input and output are expected inside the shared volume at /work
    in_name = os.path.basename(input_path)
    out_name = os.path.basename(output_path)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{vol}:/work",
        "jrottenberg/ffmpeg:4.4-alpine",
        "ffmpeg", "-y", "-i", f"/work/{in_name}", "-vf", f"scale={iw}:{ih}", f"/work/{out_name}"
    ]
    subprocess.run(cmd, capture_output=True)

###############################################################

# EX 4 - Serpentine pattern

def zig_zag_index(k, n):
    # upper side of interval
    if k >= n * (n + 1) // 2:
        i, j = zig_zag_index(n * n - 1 - k, n)
        return n - 1 - i, n - 1 - j
    # lower side of interval
    i = int((np.sqrt(1 + 8 * k) - 1) / 2)
    j = k - i * (i + 1) // 2
    return (j, i - j) if i & 1 else (i - j, j)

def serpentine(matrix):
    n = matrix.shape[0] 
    result = []
    for k in range(n * n):
        i, j = zig_zag_index(k, n)
        result.append(matrix[i, j])
    return result

###############################################################

# EX 5 - Black and white conversion with max compression

def black_and_white_max_compression(input_path: str, output_path: str):
    """Convert image to black and white with maximum compression"""
    vol = os.environ.get("SHARED_VOLUME", "practice1_shared")
    in_name = os.path.basename(input_path)
    out_name = os.path.basename(output_path)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{vol}:/work",
        "jrottenberg/ffmpeg:4.4-alpine",
        "ffmpeg", "-y", "-i", f"/work/{in_name}", "-vf", "format=gray", "-q:v", "31", f"/work/{out_name}"
    ]
    subprocess.run(cmd, capture_output=True)

def RLE(st: str) -> str:
    """Run-Length Encoding"""
    n = len(st)
    i = 0
    result = ""
    while i < n:
        count = 1
        while i < n - 1 and st[i] == st[i + 1]:
            count += 1
            i += 1
        result += st[i] + str(count)
        i += 1
    return result

###############################################################

# EX 6 - DCT (Discrete Cosine Transform)

class DCT:
    @staticmethod
    def encode(array: List) -> List:
        """Encode array using DCT"""
        array = np.array(array, dtype=float)
        result = array.copy()
        for line in range(result.ndim):
            result = dct(result, axis=line, norm="ortho")
        return result.tolist()

    @staticmethod
    def decode(array: List) -> List:
        """Decode array using inverse DCT"""
        array = np.array(array, dtype=float)
        result = array.copy()
        for line in reversed(range(result.ndim)):
            result = idct(result, axis=line, norm="ortho")
        return result.tolist()

###############################################################

# EX 7 - DWT (Discrete Wavelet Transform)

class encoderDWT:
    @staticmethod
    def encodeDWT(array: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Encode using DWT"""
        coeffs2 = pywt.dwt2(array, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        return LL, (LH, HL, HH)

    @staticmethod
    def decodeDWT(cA: np.ndarray, cD: Tuple) -> np.ndarray:
        """Decode using inverse DWT"""
        reconstructed = pywt.idwt2((cA, cD), 'bior1.3')
        return reconstructed

###############################################################

# API Endpoints

@app.get("/")
def root():
    return {"message": "practice1 API is running"}

# FFmpeg endpoint
@app.get("/ffmpeg/version")
def ffmpeg_version():
    """Get FFmpeg version"""
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    return {"ffmpeg_version": result.stdout.split('\n')[0]}

# RGB to YUV conversion endpoint
@app.post("/color/rgb-to-yuv")
def convert_rgb_to_yuv(request: ColorConversionRequest):
    """Convert RGB color to YUV"""
    y, u, v = rgb_to_yuv(request.r, request.g, request.b)
    return {"Y": y, "U": u, "V": v}

# YUV to RGB conversion endpoint
@app.post("/color/yuv-to-rgb")
def convert_yuv_to_rgb(request: YUVConversionRequest):
    """Convert YUV color to RGB"""
    r, g, b = yuv_to_rgb(request.Y, request.U, request.V)
    return {"R": r, "G": g, "B": b}

# RLE encoding endpoint
@app.post("/encoding/rle")
def encode_rle(data: dict):
    """Run-Length Encoding"""
    encoded = RLE(data.get("string", ""))
    return {"encoded": encoded}

# DCT encoding endpoint
@app.post("/transform/dct-encode")
def dct_encode(data: dict):
    """DCT Encoding"""
    array = data.get("array", [])
    encoded = DCT.encode(array)
    return {"encoded": encoded}

# DCT decoding endpoint
@app.post("/transform/dct-decode")
def dct_decode(data: dict):
    """DCT Decoding"""
    array = data.get("array", [])
    decoded = DCT.decode(array)
    return {"decoded": decoded}

# Serpentine endpoint
@app.post("/transform/serpentine")
def apply_serpentine(data: dict):
    """Apply serpentine pattern to 2D array"""
    array = np.array(data.get("array", []))
    if array.ndim != 2:
        return {"error": "Input must be a 2D array"}
    result = serpentine(array)
    return {"serpentine": result}


# Image resize endpoint (uses ffmpeg)
@app.post("/image/resize")
async def api_resize(file: UploadFile = File(...), width: int = 100, height: int = 100):
    """Resize uploaded image using ffmpeg. Returns the resized image bytes."""
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in" + suffix)
        out_path = os.path.join(td, "out" + suffix)
        content = await file.read()
        with open(in_path, "wb") as f:
            f.write(content)
        cmd = f"ffmpeg -y -i {in_path} -vf scale={width}:{height} {out_path}"
        subprocess.run(cmd, shell=True, capture_output=True)
        with open(out_path, "rb") as f:
            data = f.read()
    return Response(content=data, media_type=file.content_type or "image/jpeg")


# Black and white (max compression) endpoint
@app.post("/image/blackwhite")
async def api_blackwhite(file: UploadFile = File(...)):
    """Convert uploaded image to black & white with max compression using ffmpeg."""
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in" + suffix)
        out_path = os.path.join(td, "out.jpg")
        content = await file.read()
        with open(in_path, "wb") as f:
            f.write(content)
        cmd = f"ffmpeg -y -i {in_path} -vf format=gray -q:v 31 {out_path}"
        subprocess.run(cmd, shell=True, capture_output=True)
        with open(out_path, "rb") as f:
            data = f.read()
    return Response(content=data, media_type="image/jpeg")


# DWT endpoint: returns base64 PNGs of LL, LH, HL, HH components
@app.post("/image/dwt")
async def api_dwt(file: UploadFile = File(...)):
    """Apply 2D DWT to the uploaded image and return the four components as base64 PNGs."""
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in" + suffix)
        content = await file.read()
        with open(in_path, "wb") as f:
            f.write(content)

        # read image into numpy array
        img = mpimg.imread(in_path)
        # If image has alpha channel or 3 channels, convert to grayscale
        if img.ndim == 3:
            # if floats in [0,1], scale up
            if img.dtype == float or img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            # average channels
            gray = np.mean(img[..., :3], axis=2).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        # perform DWT
        coeffs2 = pywt.dwt2(gray, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2

        def arr_to_base64_png(arr: np.ndarray) -> str:
            # normalize to 0-255 for visualization
            a = arr.copy()
            a = a - a.min()
            if a.max() != 0:
                a = (a / a.max() * 255.0)
            a = np.clip(a, 0, 255).astype(np.uint8)
            im = Image.fromarray(a)
            bio = BytesIO()
            im.save(bio, format="PNG")
            return base64.b64encode(bio.getvalue()).decode('ascii')

        return {
            "LL": arr_to_base64_png(LL),
            "LH": arr_to_base64_png(LH),
            "HL": arr_to_base64_png(HL),
            "HH": arr_to_base64_png(HH),
        }


# Help endpoint listing important routes
@app.get("/help")
def help_routes():
    return {
        "routes": [
            {"path": "/", "method": "GET", "desc": "Root message"},
            {"path": "/ffmpeg/version", "method": "GET", "desc": "FFmpeg version"},
            {"path": "/color/rgb-to-yuv", "method": "POST", "desc": "JSON {r,g,b} -> YUV"},
            {"path": "/encoding/rle", "method": "POST", "desc": "JSON {string: '...'} -> RLE"},
            {"path": "/image/resize", "method": "POST", "desc": "form file + width + height -> resized image"},
            {"path": "/image/blackwhite", "method": "POST", "desc": "form file -> black & white compressed image"},
            {"path": "/image/dwt", "method": "POST", "desc": "form file -> base64 PNGs of LL,LH,HL,HH"}
        ]
    }