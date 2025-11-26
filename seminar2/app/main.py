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
from fastapi.responses import FileResponse


app = FastAPI()


###############################################################
#https://creatomate.com/blog/how-to-change-the-resolution-of-a-video-using-ffmpeg
# EX 3 - Resize video with ffmpeg

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
        "ffmpeg", "-i", f"/work/{in_name}", "-vf", f"scale={iw}:{ih}", f"/work/{out_name}"
    ]
    subprocess.run(cmd, capture_output=True)

###############################################################

###############################################################
#https://trac.ffmpeg.org/wiki/Chroma%20Subsampling
# EX 2 -chroma_subsambpling

def chroma_subsampling(input_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "format=yuv422p",
        "-c:v", "libx264",
        output_path
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


###############################################################

###############################################################
#https://trac.ffmpeg.org/wiki/Chroma%20Subsampling
# EX 3 - video info

def video_info(input_path: str):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_path   
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

###############################################################


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


# Image resize endpoint (uses ffmpeg)
@app.post("/video/resize")
async def api_video_resize(file: UploadFile = File(...), width: int = 640, height: int = 360):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    td = tempfile.mkdtemp()

    in_path = os.path.join(td, "in" + suffix)
    out_path = os.path.join(td, "out" + suffix)

    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)

    cmd = ["ffmpeg", "-y", "-i", in_path, "-vf", f"scale={width}:{height}", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {"error": result.stderr}

    return FileResponse(out_path, media_type="video/mp4", filename="resized.mp4")


@app.post("/video/info")
async def api_video_info(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in" + suffix)
            content = await file.read()
            with open(in_path, "wb") as f:
                f.write(content)
            info_json = video_info(in_path)
        return Response(content=info_json, media_type="application/json")
    except Exception as e:
        return {"error": str(e)}


@app.post("/video/chroma")
async def api_chroma_subsampling(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    td = tempfile.mkdtemp()  # ← DO NOT auto-delete

    in_path = os.path.join(td, "in" + suffix)
    out_path = os.path.join(td, "out" + suffix)

    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)

    result = chroma_subsampling(in_path, out_path)

    if result.returncode != 0:
        return {"error": result.stderr}

    return FileResponse(out_path, media_type="video/mp4", filename="chroma_video.mp4")



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