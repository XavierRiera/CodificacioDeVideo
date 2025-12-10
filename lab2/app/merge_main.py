# LAB 2 and SEMI 2 done with Claude Sonnet 4.2 to reduce lines of the code and have less redundancy

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Video Processing API", version="2.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

###############################################################
# UTILITIES
###############################################################

@contextmanager
def temp_workspace():
    """Context manager for temporary directories"""
    td = tempfile.mkdtemp()
    try:
        yield Path(td)
    finally:
        pass  # Cleanup handled by system


def run_ffmpeg(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Unified FFmpeg command runner with error handling"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {result.stderr}")
    return result


async def save_upload(file: UploadFile, path: Path) -> Path:
    """Save uploaded file to path"""
    full_path = path / file.filename
    content = await file.read()
    full_path.write_bytes(content)
    return full_path


###############################################################
# CORE VIDEO FUNCTIONS
###############################################################

def resize_video(input_path: Path, width: int, height: int, output_path: Path):
    """Resize video using FFmpeg"""
    cmd = ["ffmpeg", "-y", "-i", str(input_path), 
           "-vf", f"scale={width}:{height}", str(output_path)]
    run_ffmpeg(cmd)


def chroma_subsampling(input_path: Path, output_path: Path):
    """Apply chroma subsampling"""
    cmd = ["ffmpeg", "-y", "-i", str(input_path),
           "-vf", "format=yuv422p", "-c:v", "libx264", str(output_path)]
    run_ffmpeg(cmd)


def get_video_info(input_path: Path) -> str:
    """Get video metadata using ffprobe"""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(input_path)]
    return run_ffmpeg(cmd).stdout


def create_bbb_container(input_path: Path, output_path: Path):
    """Create BBB container with multiple audio tracks"""
    with temp_workspace() as td:
        clip = td / "clip_20s.mp4"
        wav = td / "audio.wav"
        
        # Extract 20 second clip and audio
        run_ffmpeg(["ffmpeg", "-i", str(input_path), "-ss", "00:00", "-to", "00:20", str(clip)])
        run_ffmpeg(["ffmpeg", "-y", "-i", str(clip), "-q:a", "0", "-map", "a", str(wav)])
        
        # Encode to different formats
        audio_tracks = {
            "aac": (td / "audio.aac", ["-c:a", "aac", "-b:a", "128k"]),
            "mp3": (td / "audio.mp3", ["-ac", "2", "-b:a", "192k"]),
            "ac3": (td / "audio.ac3", ["-c:a", "ac3", "-b:a", "192k"])
        }
        
        for path, codec_opts in audio_tracks.values():
            run_ffmpeg(["ffmpeg", "-i", str(wav)] + codec_opts + [str(path)])
        
        # Multiplex all tracks
        cmd = ["ffmpeg", "-y", "-i", str(clip)]
        for path, _ in audio_tracks.values():
            cmd.extend(["-i", str(path)])
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-map", "3:a:0",
                   "-c:v", "copy", "-c:a", "copy", str(output_path)])
        run_ffmpeg(cmd)


def count_tracks(input_path: Path) -> int:
    """Count MP4 tracks"""
    return input_path.read_bytes().count(b"trak")


def add_macroblocks_visualization(input_path: Path, output_path: Path):
    """Visualize macroblocks and motion vectors"""
    cmd = ["ffmpeg", "-flags2", "+export_mvs", "-i", str(input_path),
           "-vf", "codecview=mv=pf+bf+bb", "-c:v", "libx264", str(output_path)]
    run_ffmpeg(cmd)


def create_yuv_histogram(input_path: Path, output_path: Path):
    """Generate YUV histogram visualization"""
    cmd = ["ffmpeg", "-i", str(input_path),
           "-vf", "histogram=display_mode=stack,scale=1280:720,setsar=1",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(output_path)]
    run_ffmpeg(cmd)


###############################################################
# CODEC CONVERSION
###############################################################

CODEC_CONFIGS = {
    0: {"ext": "webm", "cmd": ["-c:v", "libvpx-vp9", "-b:v", "2M", "-c:a", "libopus"]},
    1: {"ext": "webm", "cmd": ["-c:v", "libvpx", "-b:v", "1M", "-c:a", "libvorbis"]},
    2: {"ext": "mp4", "cmd": ["-c:v", "libx265", "-vtag", "hvc1", "-c:a", "aac"]},
    3: {"ext": "mp4", "cmd": ["-c:v", "libaom-av1", "-crf", "30", "-c:a", "aac"]}
}


def convert_codec(input_path: Path, format_id: int, output_path: Path):
    """Convert video to specified codec"""
    with temp_workspace() as td:
        clip = td / "clip20.mp4"
        run_ffmpeg(["ffmpeg", "-i", str(input_path), "-ss", "00:00", "-to", "00:20", str(clip)])
        
        config = CODEC_CONFIGS.get(format_id)
        if not config:
            raise HTTPException(status_code=400, detail="Invalid format")
        
        cmd = ["ffmpeg", "-y", "-i", str(clip)] + config["cmd"] + [str(output_path)]
        run_ffmpeg(cmd)


def create_encoding_ladder(input_path: Path, output_dir: Path):
    """Generate multi-resolution encoding ladder"""
    ladder_specs = [
        ("360p_vp9.webm", 640, 360, 0),
        ("540p_vp8.webm", 960, 540, 1),
        ("720p_h265.mp4", 1280, 720, 2),
        ("1080p_av1.mp4", 1920, 1080, 3)
    ]
    
    for filename, width, height, codec in ladder_specs:
        scaled = output_dir / f"scaled_{width}x{height}.mp4"
        resize_video(input_path, width, height, scaled)
        convert_codec(scaled, codec, output_dir / filename)


###############################################################
# API ENDPOINTS
###############################################################

@app.get("/")
def root():
    return {"message": "Video Processing API", "version": "2.0"}


@app.get("/ffmpeg/version")
def ffmpeg_version():
    result = run_ffmpeg(["ffmpeg", "-version"], check=False)
    return {"version": result.stdout.split("\n")[0]}


@app.post("/video/resize", response_class=FileResponse)
async def api_resize(file: UploadFile = File(...), width: int = 640, height: int = 360):
    with temp_workspace() as td:
        input_path = await save_upload(file, td)
        output_path = td / "resized.mp4"
        resize_video(input_path, width, height, output_path)
        return FileResponse(output_path, media_type="video/mp4")


@app.post("/video/chroma", response_class=FileResponse)
async def api_chroma(file: UploadFile = File(...)):
    with temp_workspace() as td:
        input_path = await save_upload(file, td)
        output_path = td / "chroma.mp4"
        chroma_subsampling(input_path, output_path)
        return FileResponse(output_path, media_type="video/mp4")


@app.post("/video/info", response_class=JSONResponse)
async def api_info(file: UploadFile = File(...)):
    with temp_workspace() as td:
        input_path = await save_upload(file, td)
        return JSONResponse(content=get_video_info(input_path), media_type="application/json")


@app.post("/video/bbb-container", response_class=FileResponse)
async def api_bbb(file: UploadFile = File(...)):
    with temp_workspace() as td:
        input_path = await save_upload(file, td)
        output_path = td / "bbb_final.mp4"
        create_bbb_container(input_path, output_path)
        return FileResponse(output_path, media_type="video/mp4")


@app.post("/video/tracks")
async def api_tracks(file: UploadFile = File(...)):
    with temp_workspace() as td:
        input_path = await save_upload(file, td)
        return {"tracks": count_tracks(input_path)}


@app.post("/video/macroblocks", response_class=FileResponse)
async def api_macroblocks(file: UploadFile = File(...)):
    with temp_workspace() as td:
        input_path = await save_upload(file, td)
        output_path = td / "macroblocks.mp4"
        add_macroblocks_visualization(input_path, output_path)
        return FileResponse(output_path, media_type="video/mp4")


@app.post("/video/yuv-histogram", response_class=FileResponse)
async def api_histogram(file: UploadFile = File(...)):
    with temp_workspace() as td:
        input_path = await save_upload(file, td)
        output_path = td / "histogram.mp4"
        create_yuv_histogram(input_path, output_path)
        return FileResponse(output_path, media_type="video/mp4")


@app.post("/video/convert", response_class=FileResponse)
async def api_convert(file: UploadFile = File(...), format: int = 0):
    with temp_workspace() as td:
        input_path = await save_upload(file, td)
        ext = CODEC_CONFIGS.get(format, {}).get("ext", "mp4")
        output_path = td / f"output.{ext}"
        convert_codec(input_path, format, output_path)
        return FileResponse(output_path)


@app.post("/video/encoding-ladder")
async def api_ladder(file: UploadFile = File(...)):
    with temp_workspace() as td:
        input_path = await save_upload(file, td)
        create_encoding_ladder(input_path, td)
        return {"message": "Encoding ladder completed", "folder": str(td)}


@app.get("/gui", response_class=HTMLResponse)
def gui():
    gui_path = Path("app/gui_bona.html")
    if gui_path.exists():
        return HTMLResponse(gui_path.read_text())
    return HTMLResponse("<h1>GUI not found</h1>", status_code=404)
