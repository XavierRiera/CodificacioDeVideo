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

    # to VP9 From: https://trac.ffmpeg.org/wiki/Encode/VP9
    if (format == 0):
        cmd = ["ffmpeg", "-i", input_path, "-c:v", "libvpx-vp9", "-b:v", "2M", "-c:a", "libopus", output_path]

    # to VP8 From: https://trac.ffmpeg.org/wiki/Encode/VP8
    elif (format == 1):
        cmd = ["ffmpeg", "-i", input_path, "-c:v", "libvpx", "-b:v", "1M", "-c:a", "libvorbis", output_path]

    # to h265 From: https://stackoverflow.com/questions/58742765/convert-videos-from-264-to-265-hevc-with-ffmpeg
    elif (format == 2):
        cmd = ["ffmpeg", "-i", input_path, "-c:v", "libx265", "-vtag", "hvc1", output_path]

    # to AV1 From: https://trac.ffmpeg.org/wiki/Encode/AV1 
    elif (format == 3):
        cmd = ["ffmpeg", "-i", input_path, "-c:v", "libaom-av1", "-crf", "30", output_path]
    
    return subprocess.run(cmd, capture_output=True)

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
# EX 4 - new BBB container
def new_BBB_container(input_path: str, output_path: str):
    td = tempfile.mkdtemp()

    twenty_seconds = os.path.join(td, "clip_20s.mp4")
    wav = os.path.join(td, "audio.wav")
    aac_mono = os.path.join(td, "audio_aac_mono.m4a")
    mp3_stereo = os.path.join(td, "audio_mp3_stereo.mp3")
    ac3_file = os.path.join(td, "audio_ac3.ac3")

    # From: https://creatomate.com/blog/how-to-trim-a-video-using-a-start-and-stop-time-with-ffmpeg
    # Trim to 20 s
    cmd_cut = ["ffmpeg", "-i", input_path, "-ss", "00:00", "-to", "00:20", twenty_seconds]
    r = subprocess.run(cmd_cut, capture_output=True, text=True)

    # From: https://www.mux.com/articles/extract-audio-from-a-video-file-with-ffmpeg
    # Extract WAV
    cmd_wav = ["ffmpeg", "-y", "-i", twenty_seconds, "-q:a", "0", "-map", "a", wav]
    r = subprocess.run(cmd_wav, capture_output=True, text=True)

    # From: https://superuser.com/questions/684955/converting-audio-to-aac-using-ffmpeg
    # WAV to AAC mono
    cmd_aac = ["ffmpeg", "-i", wav, "-strict", "experimental", "-c:a", "aac", "-b:a", "128k", aac_mono]
    r = subprocess.run(cmd_aac, capture_output=True, text=True)
    
    # From: https://www.cincopa.com/learn/ffmpeg-for-audio-encoding-filtering-and-normalization
    # WAV to MP3 stereo (96 kbps)
    cmd_mp3 = ["ffmpeg", "-i", wav, "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", mp3_stereo]
    r = subprocess.run(cmd_mp3, capture_output=True, text=True)

    #From: https://superuser.com/questions/1279589/using-ffmpeg-to-convert-to-ac3-and-remove-extra-audio-tracks
    # WAV to AC3 (192 kbps)
    cmd_ac3 = ["ffmpeg", "-i", wav, "-c:a", "ac3", "-b:a", "192k", ac3_file]
    r = subprocess.run(cmd_ac3, capture_output=True, text=True)

    # 7) Package into final MP4: video from clip + three AAC audio tracks
    cmd_package = [
        "ffmpeg", "-y",
        "-i", twenty_seconds,
        "-i", aac_mono,
        "-i", mp3_stereo,
        "-i", ac3_file,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-map", "2:a:0",
        "-map", "3:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-metadata:s:a:0", "title=AAC_mono",
        "-metadata:s:a:1", "title=MP3_stereo_transcoded",
        "-metadata:s:a:2", "title=AC3_transcoded",
        output_path
    ]
    r = subprocess.run(cmd_package, capture_output=True, text=True)
    # also keep produced audio files next to final mp4 for inspection if caller wants them
    return r

###############################################################
# EX 5 - count tracks
# From: https://dev.to/enter?state=new-user&bb=239338
def count(input_path: str):
    with open(input_path, 'rb') as f:
        data = f.read()
        # Count how many times 'trak' appears
        return data.count(b'trak')      # literal conta quantes vegades apareix 'trak' al fitxer

##############################################################
# EX 6 - macroblocks and motion vectors
# From: https://trac.ffmpeg.org/wiki/Debug/MacroblocksAndMotionVectors?utm_source=chatgpt.com
def macroblocks_and_motion_vectors(input_path: str, output_path: str):
    cmd = ["ffmpeg", "-flags2", "+export_mvs", "-i", input_path, "-vf", "codecview=mv=pf+bf+bb", "-c:v", "libx264", output_path]

    return subprocess.run(cmd, capture_output=True, text=True)

##############################################################
# EX 7 - YUV histogram
# From: https://hhsprings.bitbucket.io/docs/programming/examples/ffmpeg/video_data_visualization/histogram.html?utm_source=chatgpt.com 

def YUV_histogram(input_path: str, output_path: str):
    cmd = ["ffmpeg", "-i", input_path, "-vf", "histogram=display_mode=stack,scale=1280:720,setsar=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", output_path]

    return subprocess.run(cmd, capture_output=True, text=True)

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

@app.post("/video/bbb-container")
async def api_create_bbb_container(file: UploadFile = File(...)):
    """
    Upload Big Buck Bunny and create a 20-second MP4
    with 3 additional audio tracks (AAC mono, MP3 stereo, AC3).
    """
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    td = tempfile.mkdtemp()

    in_path = os.path.join(td, "in" + suffix)
    out_path = os.path.join(td, "bbb_final.mp4")

    # Save input BBB
    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)

    # Run the full pipeline
    result = new_BBB_container(in_path, out_path)

    if result.returncode != 0:
        return {"error": result.stderr}

    return FileResponse(out_path, media_type="video/mp4", filename="bbb_container.mp4")


@app.post("/video/tracks")
async def api_count_tracks(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    td = tempfile.mkdtemp()
    
    in_path = os.path.join(td, "in" + suffix)
    content = await file.read()
    
    with open(in_path, "wb") as f:
        f.write(content)
    
    track_count = count(in_path)
    
    return {"tracks": track_count}

@app.post("/video/macroblocks")
async def api_macroblocks_motion(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    td = tempfile.mkdtemp()
    
    in_path = os.path.join(td, "in" + suffix)
    out_path = os.path.join(td, "out" + suffix)
    
    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)
    
    result = macroblocks_and_motion_vectors(in_path, out_path)
    
    if result.returncode != 0:
        return {"error": result.stderr}
    
    return FileResponse(out_path, media_type="video/mp4", filename="macroblocks.mp4")


@app.post("/video/yuv-histogram")
async def api_yuv_histogram(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    td = tempfile.mkdtemp()
    
    in_path = os.path.join(td, "in" + suffix)
    out_path = os.path.join(td, "histogram.mp4")
    
    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)
    
    result = YUV_histogram(in_path, out_path)
    
    if result.returncode != 0:
        return {"error": result.stderr}
    
    return FileResponse(out_path, media_type="video/mp4", filename="yuv_histogram.mp4")



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