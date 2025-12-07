# MERGE DES LAB2 I SEMI2 FET AMB ES CHAT PER PODER FER TOT EN UN SOL FITXER I QUE FUNCIONI TOT

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import FileResponse, HTMLResponse
import subprocess
import numpy as np
import pywt
from scipy.fftpack import dct, idct
import tempfile
import os
from typing import List, Tuple
import matplotlib.image as mpimg
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()

###############################################################
# LAB 1 – FUNCTIONS: VIDEO RESIZE
def resize(input_path: str, iw: int, ih: int, output_path: str):
    vol = os.environ.get("SHARED_VOLUME", "practice1_shared")
    in_name = os.path.basename(input_path)
    out_name = os.path.basename(output_path)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{vol}:/work",
        "jrottenberg/ffmpeg:4.4-alpine",
        "ffmpeg", "-i", f"/work/{in_name}",
        "-vf", f"scale={iw}:{ih}",
        f"/work/{out_name}"
    ]
    subprocess.run(cmd, capture_output=True)


###############################################################
# LAB 1 – FUNCTIONS: CHROMA SUBSAMPLING
def chroma_subsampling(input_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "format=yuv422p",
        "-c:v", "libx264",
        output_path
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


###############################################################
# LAB 1 – FUNCTIONS: VIDEO INFO
def video_info(input_path: str):
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        input_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


###############################################################
# LAB 1 – FUNCTIONS: NEW BBB CONTAINER
def new_BBB_container(input_path: str, output_path: str):
    td = tempfile.mkdtemp()
    twenty_seconds = os.path.join(td, "clip_20s.mp4")
    wav = os.path.join(td, "audio.wav")
    aac = os.path.join(td, "audio_aac.m4a")
    mp3 = os.path.join(td, "audio_mp3.mp3")
    ac3 = os.path.join(td, "audio.ac3")

    subprocess.run(["ffmpeg", "-i", input_path, "-ss", "00:00", "-to", "00:20", twenty_seconds])
    subprocess.run(["ffmpeg", "-y", "-i", twenty_seconds, "-q:a", "0", "-map", "a", wav])
    subprocess.run(["ffmpeg", "-i", wav, "-c:a", "aac", "-b:a", "128k", aac])
    subprocess.run(["ffmpeg", "-i", wav, "-ac", "2", "-b:a", "192k", mp3])
    subprocess.run(["ffmpeg", "-i", wav, "-c:a", "ac3", "-b:a", "192k", ac3])

    cmd = [
        "ffmpeg", "-y",
        "-i", twenty_seconds,
        "-i", aac, "-i", mp3, "-i", ac3,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-map", "2:a:0",
        "-map", "3:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        output_path
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


###############################################################
# LAB 1 – FUNCTIONS: COUNT TRACKS
def count_tracks(input_path: str):
    with open(input_path, 'rb') as f:
        return f.read().count(b"trak")


###############################################################
# LAB 1 – MACROBLOCKS & MOTION VECTORS
def macroblocks_and_motion_vectors(input_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-flags2", "+export_mvs",
        "-i", input_path,
        "-vf", "codecview=mv=pf+bf+bb",
        "-c:v", "libx264",
        output_path
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


###############################################################
# LAB 1 – YUV HISTOGRAM
def YUV_histogram(input_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", "histogram=display_mode=stack,scale=1280:720,setsar=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an", output_path
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


###############################################################
# LAB 2 – CONVERT TO VP9 / VP8 / H265 / AV1
def convert(input_path: str, format: int, output_path: str):
    td = tempfile.mkdtemp()
    clip20 = os.path.join(td, "clip20.mp4")
    subprocess.run(["ffmpeg", "-i", input_path, "-ss", "00:00", "-to", "00:20", clip20])

    if format == 0:
        cmd = ["ffmpeg", "-y", "-i", clip20, "-c:v", "libvpx-vp9", "-b:v", "2M", "-c:a", "libopus", output_path]
    elif format == 1:
        cmd = ["ffmpeg", "-y", "-i", clip20, "-c:v", "libvpx", "-b:v", "1M", "-c:a", "libvorbis", output_path]
    elif format == 2:
        cmd = ["ffmpeg", "-y", "-i", clip20, "-c:v", "libx265", "-vtag", "hvc1", "-c:a", "aac", output_path]
    elif format == 3:
        cmd = ["ffmpeg", "-y", "-i", clip20, "-c:v", "libaom-av1", "-crf", "30", "-c:a", "aac", output_path]

    return subprocess.run(cmd, capture_output=True)


###############################################################
# LAB 2 – ENCODING LADDER
def encoding_ladder(input_path: str, output_dir: str):
    ladder = [
        ("360p_vp9.webm", 640, 360, 0),
        ("540p_vp8.mp4", 960, 540, 1),
        ("720p_h265.mp4", 1280, 720, 2),
        ("1080p_av1.mp4", 1920, 1080, 3)
    ]

    for filename, w, h, codec in ladder:
        scaled_temp = os.path.join(output_dir, f"scaled_{w}x{h}.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-vf", f"scale={w}:{h}", scaled_temp])
        out_path = os.path.join(output_dir, filename)
        convert(scaled_temp, codec, out_path)


###############################################################
# API ENDPOINTS
@app.get("/")
def root():
    return {"message": "API running"}


@app.get("/ffmpeg/version")
def ffmpeg_version():
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    return {"version": r.stdout.split("\n")[0]}

###############################################################
# -------- LAB 1 ENDPOINTS --------

@app.post("/video/resize")
async def api_resize(file: UploadFile = File(...), width: int = 640, height: int = 360):
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, file.filename)
    out_path = os.path.join(td, "resized.mp4")

    with open(in_path, "wb") as f:
        f.write(await file.read())

    result = subprocess.run(["ffmpeg", "-y", "-i", in_path, "-vf", f"scale={width}:{height}", out_path])

    if result.returncode != 0:
        return {"error": "FFmpeg failed"}

    return FileResponse(out_path, media_type="video/mp4")


@app.post("/video/chroma")
async def api_chroma(file: UploadFile = File(...)):
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, file.filename)
    out_path = os.path.join(td, "chroma.mp4")

    with open(in_path, "wb") as f:
        f.write(await file.read())

    r = chroma_subsampling(in_path, out_path)
    if r.returncode != 0:
        return {"error": r.stderr}

    return FileResponse(out_path, media_type="video/mp4")


@app.post("/video/info")
async def api_info(file: UploadFile = File(...)):
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, file.filename)

    with open(in_path, "wb") as f:
        f.write(await file.read())

    return Response(content=video_info(in_path), media_type="application/json")


@app.post("/video/bbb-container")
async def api_bbb(file: UploadFile = File(...)):
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, file.filename)
    out_path = os.path.join(td, "bbb_final.mp4")

    with open(in_path, "wb") as f:
        f.write(await file.read())

    r = new_BBB_container(in_path, out_path)
    if r.returncode != 0:
        return {"error": r.stderr}

    return FileResponse(out_path, media_type="video/mp4")


@app.post("/video/tracks")
async def api_tracks(file: UploadFile = File(...)):
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, file.filename)

    with open(in_path, "wb") as f:
        f.write(await file.read())

    return {"tracks": count_tracks(in_path)}


@app.post("/video/macroblocks")
async def api_macro(file: UploadFile = File(...)):
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, file.filename)
    out_path = os.path.join(td, "macroblocks.mp4")

    with open(in_path, "wb") as f:
        f.write(await file.read())

    r = macroblocks_and_motion_vectors(in_path, out_path)
    if r.returncode != 0:
        return {"error": r.stderr}

    return FileResponse(out_path, media_type="video/mp4")


@app.post("/video/yuv-histogram")
async def api_hist(file: UploadFile = File(...)):
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, file.filename)
    out_path = os.path.join(td, "hist.mp4")

    with open(in_path, "wb") as f:
        f.write(await file.read())

    r = YUV_histogram(in_path, out_path)
    if r.returncode != 0:
        return {"error": r.stderr}

    return FileResponse(out_path, media_type="video/mp4")


###############################################################
# -------- LAB 2 ENDPOINTS --------

@app.post("/video/convert")
async def api_convert(file: UploadFile = File(...), format: int = 0):
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, file.filename)

    if format in [0, 1]:
        out_path = os.path.join(td, "output.webm")
    else:
        out_path = os.path.join(td, "output.mp4")

    with open(in_path, "wb") as f:
        f.write(await file.read())

    r = convert(in_path, format, out_path)
    if r.returncode != 0:
        return {"error": r.stderr}

    return FileResponse(out_path)


@app.post("/video/encoding-ladder")
async def api_ladder(file: UploadFile = File(...)):
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, file.filename)

    with open(in_path, "wb") as f:
        f.write(await file.read())

    encoding_ladder(in_path, td)

    return {"message": "ladder completed", "folder": td}


###############################################################
# GUI
@app.get("/gui")
def gui():
    with open("app/gui_bona.html") as f:
        return HTMLResponse(f.read())
