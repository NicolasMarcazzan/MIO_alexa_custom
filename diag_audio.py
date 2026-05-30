#!/usr/bin/env python3
"""Quick diagnostic: check audio tools and test playback."""
import shutil
import sys

print("=== Audio diagnostics ===")
print(f"pw-play: {shutil.which('pw-play')}")
print(f"parec:   {shutil.which('parec')}")
print(f"aplay:   {shutil.which('aplay')}")

# Test aplay -D pipewire
if shutil.which("aplay"):
    import subprocess
    print("\nTesting aplay -D pipewire...")
    result = subprocess.run(
        ["aplay", "-D", "pipewire", "--list-plugs"],
        capture_output=True, text=True, timeout=5
    )
    print(f"  returncode: {result.returncode}")
    print(f"  stdout: {result.stdout.strip()!r}")
    print(f"  stderr: {result.stderr.strip()!r}")

# Test sounddevice
print("\nTesting sounddevice...")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    print(f"  Default output device: {sd.default.device}")
    for i, d in enumerate(devices):
        if d['max_output_channels'] > 0:
            print(f"  Device {i}: {d['name']} (out={d['max_output_channels']}, hostapi={d['hostapi']})")
    
    # Try playing a short tone
    import numpy as np
    rate = 48000
    duration = 0.3
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    print(f"\n  Playing 0.3s tone at 880Hz via sounddevice...")
    sd.play(tone, rate, blocking=True)
    sd.wait()
    print("  [sounddevice playback finished]")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
