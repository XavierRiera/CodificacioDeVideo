This project implements an API-based system for multimedia processing using FastAPI, Docker, and FFmpeg. The assignment required containerizing both the API and FFmpeg and integrating previous work involving image transforms, compression, color conversion, and signal-processing techniques.

The resulting API exposes a set of endpoints (instead of two as the assignmanet says, as we really liked the result and we were motivated to try more endpoints) that allow users to:
- Convert between RGB and YUV color spaces
- Apply Run-Length Encoding
- Perform DCT and DWT transformations
- Resize images using FFmpeg
- Convert images to compressed black & white format
- Apply a serpentine (zig‑zag) matrix traversal

The entire system is orchestrated with docker-compose, enabling communication between the API container and an FFmpeg container.

The main.py file integrates multiple tasks into one API.
We have two types of FFmpeg operations:
- resize images using scale filters
- convert to grayscale with max compression
FFmpeg is executed inside its own container, called via subprocess.
This fulfills the requirement of making the API interact with a separate FFmpeg Docker.
