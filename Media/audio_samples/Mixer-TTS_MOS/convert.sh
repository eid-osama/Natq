#!/bin/bash

IMAGE="ChatGPT Image Jun 9, 2025, 02_53_58 AM.png"

for AUDIO in *.wav; do
    # Create output filename by replacing .wav with .mp4
    OUTPUT="${AUDIO%.wav}.mp4"
    
    ffmpeg -loop 1 -y -i "$IMAGE" -i "$AUDIO" \
    -c:v libx264 -c:a aac -b:a 192k -shortest -movflags +faststart "$OUTPUT"
done
