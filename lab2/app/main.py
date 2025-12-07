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
# EX 1 - convert

def convert(input_path: str, format: int, output_path: str):

    # Cut first 20 seconds to speed up processing --> així no tarda sa vida
    td = tempfile.mkdtemp()
    twenty_seconds = os.path.join(td, "clip_20s.mp4")
    cmd_cut = ["ffmpeg", "-i", input_path, "-ss", "00:00", "-to", "00:20", twenty_seconds]
    subprocess.run(cmd_cut, capture_output=True, text=True)

    # to VP9 From: https://trac.ffmpeg.org/wiki/Encode/VP9
    if (format == 0):
        cmd = ["ffmpeg", "-y", "-i", twenty_seconds, "-c:v", "libvpx-vp9", "-b:v", "2M", "-c:a", "libopus", output_path]

    # to VP8 From: https://trac.ffmpeg.org/wiki/Encode/VP8
    elif (format == 1):
        cmd = ["ffmpeg", "-y", "-i", twenty_seconds, "-c:v", "libvpx", "-b:v", "1M", "-c:a", "libvorbis", output_path]

    # to h265 From: https://stackoverflow.com/questions/58742765/convert-videos-from-264-to-265-hevc-with-ffmpeg
    elif (format == 2):
        cmd = ["ffmpeg", "-y", "-i", twenty_seconds, "-c:v", "libx265", "-vtag", "hvc1", "-c:a", "aac", output_path]            # added -c:a aac for audio --> before, only audio was saved into output

    # to AV1 From: https://trac.ffmpeg.org/wiki/Encode/AV1 
    elif (format == 3):
        cmd = ["ffmpeg", "-y", "-i", twenty_seconds, "-c:v", "libaom-av1", "-crf", "30", "-c:a", "aac", output_path]            # added -c:a aac for audio --> before, only audio was saved into output
    
    return subprocess.run(cmd, capture_output=True)

###############################################################

###############################################################
#https://trac.ffmpeg.org/wiki/Chroma%20Subsampling
# EX 2 - encoding ladder

def encoding_ladder(input_path: str, output_dir: str):

    # Encodfing ladder definitions --> filename, width, height, codec
    ladder = [
        ("360p_vp9.webm", 640, 360, 0),         # VP9
        ("540p_vp8.mp4", 960, 540, 1),          # VP8
        ("720p_h265.mp4", 1280, 720, 2),        # h265
        ("1080p_av1.mp4", 1920, 1080, 3)        # AV1
    ]

    for filename, w, h, codec in ladder:
        scaled = os.path.join(output_dir, f"scaled_{w}x{h}.mp4")                        # temporary scaled file

        # 1. Resize using ffmpeg
        # From: https://trac.ffmpeg.org/wiki/Scaling
        scale_cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", f"scale={w}:{h}", scaled]       # scale command
        subprocess.run(scale_cmd, capture_output=True, text=True)                       # run scaling

        # 2. Convert using EX1
        out_path = os.path.join(output_dir, filename)                                   # final output path
        convert(scaled, codec, out_path)                                                # run convert to target codec

#############################################################

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


@app.post("/video/convert")
async def api_convert(file: UploadFile = File(...), format: int = 0):
    # format: 0=VP9, 1=VP8, 2=h265, 3=AV1
    
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    td = tempfile.mkdtemp()
    
    in_path = os.path.join(td, "in" + suffix)
    
    # Set output extension based on format
    if format == 0:
        out_path = os.path.join(td, "output.webm")  # VP9 usually in webm
    elif format == 1:
        out_path = os.path.join(td, "output.webm")  # VP8 usually in webm
    elif format == 2:
        out_path = os.path.join(td, "output.mp4")   # h265 in mp4
    elif format == 3:
        out_path = os.path.join(td, "output.mp4")   # AV1 in mp4
    else:
        return {"error": "Invalid format"}
    
    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)
    
    result = convert(in_path, format, out_path)
    
    if result.returncode != 0:
        return {"error": result.stderr}
    
    # Return appropriate content type
    if format == 0 or format == 1:
        return FileResponse(out_path, media_type="video/webm", filename="converted.webm")
    else:
        return FileResponse(out_path, media_type="video/mp4", filename="converted.mp4")


@app.post("/video/encoding-ladder")
async def api_encoding_ladder(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    td = tempfile.mkdtemp()

    in_path = os.path.join(td, "input" + suffix)

    # Save uploaded file
    with open(in_path, "wb") as f:
        f.write(await file.read())

    # Run ladder
    encoding_ladder(in_path, td)

    return {
        "message": "encoding ladder completed",
        "folder": td
    }


from fastapi.responses import HTMLResponse

@app.get("/gui")
def gui():
    with open("app/gui.html") as f:
        return HTMLResponse(f.read())

