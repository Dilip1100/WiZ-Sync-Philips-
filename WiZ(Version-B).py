#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WiZ Batten Local Colour Sync - Python 3.12
Screen colour sampling with automatic local-network discovery.
"""

import argparse
import ctypes
import importlib.util
import ipaddress
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import keyboard
import numpy as np
from PIL import ImageCms

# ------------------- 3rd party packages -------------------
import mss                     # pip install mss
from skimage import color as skcolor   # pip install scikit-image

# ------------------- CONFIG -------------------
PORT = 38899
REFRESH_RATE_HZ = 74
SLEEP_INTERVAL = 1.0 / REFRESH_RATE_HZ

BRIGHTNESS_MIN = 35
BRIGHTNESS_MAX = 100
BRIGHTNESS_STEP = 10
BRIGHTNESS_OFFSET_MIN = -30
BRIGHTNESS_OFFSET_MAX = 30

SATURATION_BOOST = 1.50          # ↑ from 1.12 — stronger vivid chroma push
DARK_LSTAR_THRESHOLD = 6
FALLBACK_RGB = (8, 7, 6)
NEUTRAL_RGB = (255, 244, 230)
MIN_PIXEL_SATURATION = 12        # ↓ from 24 — catch subtler colours
MIN_PIXEL_VALUE = 12             # ↓ from 18 — include dark saturated pixels
MIN_COLOR_CONFIDENCE = 0.015     # ↓ from 0.035 — less filtering
PURPLE_HUE_MIN = 265
PURPLE_HUE_MAX = 315
PURPLE_CONFIDENCE_MIN = 0.18
SCREEN_GAMMA = 2.2
LIGHT_GAMMA = 1.00               # ↑ from 0.92 — no gamma shift, linear passthrough
LIGHT_CHANNEL_GAINS = np.array([1.00, 1.00, 1.00], dtype=np.float32)  # ← removed channel bias
HIGHLIGHT_TRIM_PERCENTILE = 99.2
SHADOW_TRIM_PERCENTILE = 0.4
VIBRANT_MIX_MAX = 0.65           # ↑ from 0.22 — vibrant extraction dominates the blend
ZONE_WEIGHTS = np.array([1.00, 1.00, 1.35, 1.00, 1.00], dtype=np.float64)

ZONE_CFG = [
    (0.0, 0.0, "TL"),
    (0.9, 0.0, "TR"),
    (0.45, 0.45, "C"),
    (0.0, 0.9, "BL"),
    (0.9, 0.9, "BR"),
]
ZONE_SIZE_PX = 120
DEBUG_VIEW = False

# ------------------- AUTO DISCOVERY CONFIG -------------------
DISCOVERY_TIMEOUT = 1.5
BROADCAST_ADDR = "255.255.255.255"

# ------------------- LOGGING -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------- GLOBALS -------------------
DEFAULT_IP_ADDRESS = "192.168.1.3"
APP_NAME = "WiZSyncKK"
APP_USER_MODEL_ID = "dilip.WiZSyncKK.Control"
IP_ADDRESS: str = DEFAULT_IP_ADDRESS
power_state = True
brightness_level = BRIGHTNESS_MAX
brightness_offset = 0
manual_brightness_control = False
sync_active = True
last_color = np.array([255, 255, 255], dtype=np.uint8)
use_vivid_mode = True
light_gamma = LIGHT_GAMMA
light_channel_gains = LIGHT_CHANNEL_GAINS.copy()
last_status_at = 0.0

SMOOTHING_ALPHA = 0.55           # ↑ from 0.30 — faster colour tracking
KEY_DEBOUNCE_SECONDS = 0.22
STATUS_INTERVAL_SECONDS = 2.0
INTERP_STEPS = max(1, int(0.30 * REFRESH_RATE_HZ))
interp_queue: deque = deque(maxlen=INTERP_STEPS)
last_keypress_at = {
    "up": 0.0,
    "down": 0.0,
    "space": 0.0,
    "v": 0.0,
}

# ------------------- ICC HELPERS -------------------
def get_monitor_profile() -> Optional[ImageCms.ImageCmsProfile]:
    try:
        return ImageCms.get_display_profile()
    except Exception:
        return None

monitor_profile = get_monitor_profile()
srgb_profile = ImageCms.createProfile("sRGB")

# ------------------- SCREEN CAPTURE -------------------
screen_capture = threading.local()

def get_screen_capture() -> mss.mss:
    if not hasattr(screen_capture, "instance"):
        screen_capture.instance = mss.mss()
    return screen_capture.instance

def capture_zone(rel_x: float, rel_y: float, size: int, screen_w: int, screen_h: int) -> np.ndarray:
    px = min(max(0, int(rel_x * screen_w)), max(0, screen_w - size))
    py = min(max(0, int(rel_y * screen_h)), max(0, screen_h - size))
    monitor = {"top": py, "left": px, "width": size, "height": size}
    img = np.array(get_screen_capture().grab(monitor))[:, :, :3]
    return img

# ------------------- COLOUR CONVERSION -------------------
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb_float = rgb.astype(np.float32) / 255.0
    return skcolor.rgb2lab(rgb_float)

def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    rgb_float = skcolor.lab2rgb(lab)
    return np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)

def hsv_to_rgb(hue_deg: float, saturation: float, value: float) -> np.ndarray:
    hsv = np.array([[[hue_deg / 2.0, saturation, value]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]

def rgb_to_hsv_degrees(rgb: np.ndarray) -> Tuple[float, float, float]:
    hsv = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    return float(hsv[0]) * 2.0, float(hsv[1]), float(hsv[2])

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.power(srgb, SCREEN_GAMMA)

def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    linear = np.clip(linear_rgb.astype(np.float32), 0.0, 1.0)
    srgb = np.power(linear, 1.0 / SCREEN_GAMMA)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)

def weighted_linear_mean(zone_rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(zone_rgb)
    luma = (
        linear[:, :, 0] * 0.2126
        + linear[:, :, 1] * 0.7152
        + linear[:, :, 2] * 0.0722
    )
    low, high = np.percentile(luma, [SHADOW_TRIM_PERCENTILE, HIGHLIGHT_TRIM_PERCENTILE])
    trim_mask = (luma >= low) & (luma <= high)
    if not np.any(trim_mask):
        trim_mask = np.ones(luma.shape, dtype=bool)

    hsv = cv2.cvtColor(zone_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    colour_weight = 1.0 + 0.35 * saturation + 0.12 * value
    colour_weight = np.where(trim_mask, colour_weight, 0.0)

    total = float(colour_weight.sum())
    if total <= 0:
        return linear.reshape(-1, 3).mean(axis=0)
    return (linear * colour_weight[:, :, None]).sum(axis=(0, 1)) / total

def circular_hue_mean(hues_deg: np.ndarray, weights: np.ndarray) -> float:
    angles = np.deg2rad(hues_deg)
    x = np.sum(np.cos(angles) * weights)
    y = np.sum(np.sin(angles) * weights)
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)

def zone_vibrant_colour(zone_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    hsv = cv2.cvtColor(zone_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    color_mask = (saturation >= MIN_PIXEL_SATURATION) & (value >= MIN_PIXEL_VALUE)
    if not np.any(color_mask):
        return zone_rgb.mean(axis=(0, 1)).astype(np.uint8), 0.0

    hue = hue[color_mask]
    saturation = saturation[color_mask]
    value = value[color_mask]
    pixel_weights = (saturation / 255.0) ** 1.7 * (value / 255.0) ** 1.2

    # Guard: if all weights are zero (shouldn't happen after mask, but be safe)
    if pixel_weights.sum() <= 0:
        return zone_rgb.mean(axis=(0, 1)).astype(np.uint8), 0.0

    strongest = np.argsort(pixel_weights)[-max(16, len(pixel_weights) // 4):]
    pixel_weights = pixel_weights[strongest]

    hue_deg = circular_hue_mean(hue[strongest], pixel_weights)
    sat = float(np.average(saturation[strongest], weights=pixel_weights))
    val = float(np.average(value[strongest], weights=pixel_weights))
    confidence = float(np.clip(np.mean(pixel_weights) * (len(strongest) / zone_rgb[:, :, 0].size), 0.0, 1.0))
    return hsv_to_rgb(hue_deg, sat, val), confidence

def zone_screen_colour(zone_rgb: np.ndarray) -> Tuple[np.ndarray, float, float]:
    base_linear = weighted_linear_mean(zone_rgb)
    base_rgb = linear_to_srgb(base_linear)
    vibrant_rgb, vibrant_confidence = zone_vibrant_colour(zone_rgb)

    hsv = cv2.cvtColor(zone_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    colour_mask = (saturation >= MIN_PIXEL_SATURATION) & (value >= MIN_PIXEL_VALUE)
    colour_confidence = float(np.clip(
        np.mean(saturation[colour_mask] / 255.0) * np.mean(value[colour_mask] / 255.0)
        if np.any(colour_mask) else 0.0,
        0.0,
        1.0,
    ))

    mix = min(VIBRANT_MIX_MAX, max(vibrant_confidence, colour_confidence) * VIBRANT_MIX_MAX)
    mixed_linear = base_linear * (1.0 - mix) + srgb_to_linear(vibrant_rgb) * mix
    mixed_rgb = linear_to_srgb(mixed_linear)
    lstar = float(rgb_to_lab(base_rgb)[0])
    return mixed_rgb, max(vibrant_confidence, colour_confidence), lstar

def suppress_purple_haze(rgb: np.ndarray, confidence: float, scene_lstar: float) -> np.ndarray:
    # Disabled: removed for precise screen colour mimicry
    return rgb

# ------------------- CORE PROCESSING -------------------
def process_zones(zones_bgr: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    rgb_zones = [cv2.cvtColor(z, cv2.COLOR_BGR2RGB) for z in zones_bgr]
    zone_samples = [zone_screen_colour(z) for z in rgb_zones]
    zone_colours = [sample[0] for sample in zone_samples]
    confidences = np.array([sample[1] for sample in zone_samples], dtype=np.float64)
    l_stars = np.array([sample[2] for sample in zone_samples], dtype=np.float64)

    light_weights = l_stars + 10.0
    light_weights = light_weights / light_weights.sum() if light_weights.sum() > 0 else np.ones_like(l_stars) / len(l_stars)
    weights = ZONE_WEIGHTS[:len(zone_colours)] * (0.70 + 0.30 * light_weights)
    weights = weights / weights.sum()

    if confidences.sum() <= MIN_COLOR_CONFIDENCE and float(np.average(l_stars, weights=weights)) < DARK_LSTAR_THRESHOLD:
        avg_rgb = np.array(FALLBACK_RGB, dtype=np.uint8)
        colour_confidence = 0.0
    else:
        zone_linear = np.stack([srgb_to_linear(c) for c in zone_colours])
        avg_rgb = linear_to_srgb(np.average(zone_linear, axis=0, weights=weights))
        colour_confidence = float(np.average(confidences, weights=weights))

    scene_lstar = float(np.average(l_stars, weights=weights))
    dimming_curve = np.clip((scene_lstar / 100.0) ** 0.72, 0.0, 1.0)
    dimming = int(BRIGHTNESS_MIN + dimming_curve * (BRIGHTNESS_MAX - BRIGHTNESS_MIN))
    dimming = np.clip(dimming, BRIGHTNESS_MIN, BRIGHTNESS_MAX)

    avg_rgb = suppress_purple_haze(avg_rgb, colour_confidence, scene_lstar)

    if use_vivid_mode and colour_confidence >= MIN_COLOR_CONFIDENCE:
        lab = rgb_to_lab(avg_rgb)
        L, a, b = lab[0], lab[1], lab[2]
        base_chroma = np.sqrt(a**2 + b**2)
        if base_chroma > 1e-6:  # guard against neutral grey divide-by-zero
            chroma = min(base_chroma * SATURATION_BOOST, 140)
            scale = chroma / base_chroma
            a *= scale
            b *= scale
            avg_rgb = lab_to_rgb(np.array([L, a, b]))

    return avg_rgb, dimming

def apply_light_calibration(rgb: np.ndarray) -> np.ndarray:
    rgb_float = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    corrected = np.power(rgb_float, light_gamma) * light_channel_gains
    return np.clip(np.rint(corrected * 255.0), 0, 255).astype(np.uint8)

# ------------------- INTERPOLATION -------------------
def smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)

def interpolate(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    interp_queue.append(target.copy())
    if len(interp_queue) == 1:
        blended_target = target
    else:
        weights = np.linspace(0.35, 1.0, len(interp_queue), dtype=np.float32)
        blended_target = np.average(np.stack(interp_queue), axis=0, weights=weights)

    eased_alpha = smoothstep(SMOOTHING_ALPHA)
    return (current * (1 - eased_alpha) + blended_target * eased_alpha).astype(np.uint8)

# ------------------- UDP COMMAND -------------------
def send_udp_command(payload: dict, retries: int = 2):
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(0.008)
                sock.sendto(data, (IP_ADDRESS, PORT))
            return
        except Exception as e:
            if attempt == retries:
                logging.error("UDP send failed: %s", e)
            else:
                time.sleep(0.005)

# ------------------- DEBUG VIEW (optional) -------------------
def draw_debug_view(zones_bgr: List[np.ndarray], final_rgb: np.ndarray):
    if not DEBUG_VIEW:
        return

    h, w = 100, 100
    canvas = np.full((h, 5 * w, 3), 30, dtype=np.uint8)

    for i, zone in enumerate(zones_bgr):
        rgb_zone = cv2.cvtColor(zone, cv2.COLOR_BGR2RGB)
        thumb = cv2.resize(rgb_zone, (w, h), interpolation=cv2.INTER_AREA)
        canvas[:, i * w:(i + 1) * w] = thumb

    bar = np.full((20, 5 * w, 3), final_rgb, dtype=np.uint8)
    canvas[0:20, :] = bar

    cv2.imshow("WiZ Colour-Sync Debug", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

# ------------------- DISCOVERY -------------------
def is_likely_lan_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False

    cgnat = ipaddress.ip_network("100.64.0.0/10")
    return (
        isinstance(ip, ipaddress.IPv4Address)
        and ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and ip not in cgnat
    )

def get_local_subnets() -> List[ipaddress.IPv4Network]:
    subnets = []
    seen = set()

    def add_subnet(ip_text: str):
        if not is_likely_lan_ip(ip_text):
            return
        subnet = ipaddress.ip_network(f"{ip_text}/24", strict=False)
        if subnet not in seen:
            seen.add(subnet)
            subnets.append(subnet)

    try:
        output = subprocess.check_output(
            ["ipconfig"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=2.0,
        )
        for line in output.splitlines():
            if "IPv4 Address" not in line:
                continue
            _, _, value = line.partition(":")
            add_subnet(value.strip().split("(")[0].strip())
    except Exception as e:
        logging.debug("Could not read adapter list from ipconfig: %s", e)

    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add_subnet(result[4][0])
    except Exception as e:
        logging.debug("Could not infer subnets from hostname: %s", e)

    return subnets

def wait_for_discovery_response(sock: socket.socket, deadline: float) -> Optional[str]:
    while time.time() < deadline:
        try:
            data, (ip, _) = sock.recvfrom(4096)
            response = json.loads(data.decode("utf-8"))
            if response.get("id") == 9999 and "result" in response:
                return ip
        except socket.timeout:
            return None
        except Exception as e:
            logging.debug("Ignoring discovery response: %s", e)
    return None

def discover_wiz_ip(scan_subnet: bool = True) -> str:
    global IP_ADDRESS
    payload = json.dumps({
        "method": "getPilot",
        "id": 9999,
        "params": {}
    }).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.12)
    deadline = time.time() + DISCOVERY_TIMEOUT

    try:
        sock.sendto(payload, (BROADCAST_ADDR, PORT))
        ip = wait_for_discovery_response(sock, deadline)
        if ip:
            logging.info(f"WiZ device discovered at {ip}")
            IP_ADDRESS = ip
            return ip

        if scan_subnet:
            subnets = get_local_subnets()
            if subnets:
                logging.info(
                    "Broadcast did not answer; scanning %s",
                    ", ".join(str(subnet) for subnet in subnets),
                )
                for subnet in subnets:
                    for host in subnet.hosts():
                        sock.sendto(payload, (str(host), PORT))
                    ip = wait_for_discovery_response(sock, time.time() + DISCOVERY_TIMEOUT)
                    if ip:
                        logging.info(f"WiZ device discovered at {ip}")
                        IP_ADDRESS = ip
                        return ip
            else:
                logging.warning("No suitable LAN subnet found for WiZ discovery scan.")

        logging.warning("Discovery timed out. Using fallback IP %s.", IP_ADDRESS)
    except Exception as e:
        logging.error(f"Discovery error: {e}")
    finally:
        sock.close()
    
    return IP_ADDRESS  # fallback

# ------------------- DEVICE INIT -------------------
def ensure_device_ready():
    logging.info("Initialising WiZ Batten...")
    for r, g, b in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        send_udp_command({
            "id": 1,
            "method": "setPilot",
            "params": {"state": True, "r": r, "g": g, "b": b, "dimming": BRIGHTNESS_MAX},
        })
        time.sleep(0.45)
    return True

# ------------------- CONTROLS / CLI -------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync a WiZ RGB light to sampled screen colours.")
    parser.add_argument("--ip", default=DEFAULT_IP_ADDRESS, help="Fallback or fixed WiZ device IP.")
    parser.add_argument("--no-discovery", action="store_true", help="Skip UDP discovery and use --ip.")
    parser.add_argument("--rate", type=int, default=REFRESH_RATE_HZ, help="Refresh rate in Hz.")
    parser.add_argument("--zone-size", type=int, default=ZONE_SIZE_PX, help="Square sample size in pixels.")
    parser.add_argument("--debug", action="store_true", help="Show the sampled zones and final colour.")
    parser.add_argument("--normal", action="store_true", help="Start without vivid colour boost.")
    parser.add_argument("--vivid", action="store_true", help="Start with vivid colour boost enabled.")
    parser.add_argument("--light-gamma", type=float, default=LIGHT_GAMMA, help="LED gamma correction. Lower lifts darker channels.")
    parser.add_argument("--red-gain", type=float, default=float(LIGHT_CHANNEL_GAINS[0]), help="Red LED calibration gain.")
    parser.add_argument("--green-gain", type=float, default=float(LIGHT_CHANNEL_GAINS[1]), help="Green LED calibration gain.")
    parser.add_argument("--blue-gain", type=float, default=float(LIGHT_CHANNEL_GAINS[2]), help="Blue LED calibration gain.")
    parser.add_argument("--no-startup-test", action="store_true", help="Skip the RGB startup flash.")
    parser.add_argument("--keep-on-exit", action="store_true", help="Leave the light on when quitting.")
    parser.add_argument("--no-gui", action="store_true", help="Run with keyboard controls only.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()

def key_pressed_once(key: str) -> bool:
    now = time.time()
    if keyboard.is_pressed(key) and now - last_keypress_at[key] >= KEY_DEBOUNCE_SECONDS:
        last_keypress_at[key] = now
        return True
    return False

def apply_brightness_offset(auto_dim: int) -> int:
    return int(np.clip(auto_dim + brightness_offset, BRIGHTNESS_MIN, BRIGHTNESS_MAX))

def print_status(rgb: np.ndarray):
    global last_status_at
    now = time.time()
    if now - last_status_at < STATUS_INTERVAL_SECONDS:
        return
    last_status_at = now
    print(
        f"[SYNC] RGB {int(rgb[0]):03d},{int(rgb[1]):03d},{int(rgb[2]):03d} | "
        f"brightness {brightness_level}% | offset {brightness_offset:+d} | "
        f"{'vivid' if use_vivid_mode else 'normal'}"
    )

def set_power_state(enabled: bool):
    global power_state
    power_state = enabled
    print(f"[POWER] {'ON' if power_state else 'OFF'}")
    if not power_state:
        send_udp_command({"id": 1, "method": "setPilot", "params": {"state": False}})

def set_manual_brightness(value: int):
    global brightness_level
    brightness_level = int(np.clip(value, BRIGHTNESS_MIN, BRIGHTNESS_MAX))
    print(f"[BRIGHTNESS] {brightness_level}%")

def resource_path(filename: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, filename)

def set_windows_taskbar_app_id():
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception as e:
        logging.debug("Could not set Windows taskbar app ID: %s", e)

def launch_gui(sync_thread: threading.Thread) -> int:
    global sync_active

    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError:
        print("[ERROR] PyQt5 is not installed. Install it with: pip install PyQt5")
        sync_active = False
        sync_thread.join(timeout=3.0)
        return 1

    set_windows_taskbar_app_id()
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)

    icon_path = resource_path("KK.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QtGui.QIcon(icon_path))

    # ---- Cyberpunk palette ----
    CP_BG        = "#0a0a12"   # deep void black
    CP_PANEL     = "#0e0e1c"   # panel surface
    CP_BORDER    = "#1a1a30"   # subtle border
    CP_CYAN      = "#00f5ff"   # neon cyan — primary accent
    CP_MAGENTA   = "#ff2d78"   # neon magenta — danger / off state
    CP_YELLOW    = "#f5e642"   # neon yellow — highlight
    CP_DIM       = "#2a2a45"   # muted element
    CP_TEXT      = "#c8d8ff"   # primary text
    CP_SUBTEXT   = "#4a5070"   # secondary text

    STYLESHEET = f"""
        /* ── Base ── */
        QWidget {{
            background: {CP_BG};
            color: {CP_TEXT};
            font-family: "Consolas", "Courier New", monospace;
            font-size: 12px;
        }}

        /* ── Header bar ── */
        QFrame#headerBar {{
            background: {CP_PANEL};
            border-bottom: 1px solid {CP_CYAN};
        }}
        QLabel#titleLabel {{
            color: {CP_CYAN};
            font-size: 22px;
            font-weight: 700;
            letter-spacing: 6px;
            background: transparent;
        }}
        QLabel#subtitleLabel {{
            color: {CP_SUBTEXT};
            font-size: 9px;
            letter-spacing: 3px;
            background: transparent;
        }}

        /* ── Section separators ── */
        QFrame#cyanLine {{
            background: {CP_CYAN};
            border: none;
            max-height: 1px;
            min-height: 1px;
        }}
        QFrame#magentaLine {{
            background: {CP_MAGENTA};
            border: none;
            max-height: 1px;
            min-height: 1px;
        }}

        /* ── Panels ── */
        QFrame#panel {{
            background: {CP_PANEL};
            border: 1px solid {CP_BORDER};
            border-left: 2px solid {CP_CYAN};
        }}

        /* ── Labels ── */
        QLabel#sectionTag {{
            color: {CP_SUBTEXT};
            font-size: 9px;
            letter-spacing: 2px;
            background: transparent;
        }}
        QLabel#valueLabel {{
            color: {CP_CYAN};
            font-size: 28px;
            font-weight: 700;
            background: transparent;
        }}
        QLabel#deviceLabel {{
            color: {CP_SUBTEXT};
            font-size: 9px;
            letter-spacing: 1px;
            background: transparent;
        }}
        QLabel#statusDot {{
            color: {CP_CYAN};
            font-size: 11px;
            background: transparent;
        }}

        /* ── Power button — ON state ── */
        QPushButton#powerBtn {{
            background: transparent;
            border: 1px solid {CP_CYAN};
            border-radius: 0px;
            color: {CP_CYAN};
            font-family: "Consolas", monospace;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 4px;
            min-height: 46px;
            padding: 0 20px;
        }}
        QPushButton#powerBtn:hover {{
            background: rgba(0, 245, 255, 0.08);
            border-color: {CP_CYAN};
            color: #ffffff;
        }}
        QPushButton#powerBtn:pressed {{
            background: rgba(0, 245, 255, 0.18);
        }}
        /* OFF state */
        QPushButton#powerBtn:!checked {{
            border-color: {CP_MAGENTA};
            color: {CP_MAGENTA};
        }}
        QPushButton#powerBtn:!checked:hover {{
            background: rgba(255, 45, 120, 0.08);
            color: #ffffff;
        }}

        /* ── Vivid toggle button ── */
        QPushButton#vividBtn {{
            background: transparent;
            border: 1px solid {CP_DIM};
            border-radius: 0px;
            color: {CP_SUBTEXT};
            font-family: "Consolas", monospace;
            font-size: 11px;
            letter-spacing: 2px;
            min-height: 32px;
            padding: 0 12px;
        }}
        QPushButton#vividBtn:checked {{
            border-color: {CP_YELLOW};
            color: {CP_YELLOW};
            background: rgba(245, 230, 66, 0.06);
        }}
        QPushButton#vividBtn:hover {{
            border-color: {CP_TEXT};
            color: {CP_TEXT};
        }}

        /* ── Slider ── */
        QSlider::groove:horizontal {{
            background: {CP_DIM};
            border: none;
            height: 4px;
            border-radius: 2px;
        }}
        QSlider::sub-page:horizontal {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {CP_MAGENTA}, stop:1 {CP_CYAN});
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {CP_BG};
            border: 2px solid {CP_CYAN};
            border-radius: 6px;
            width: 12px;
            height: 12px;
            margin: -5px 0;
        }}
        QSlider::handle:horizontal:hover {{
            background: {CP_CYAN};
            border-color: #ffffff;
        }}
    """

    class GlitchLabel(QtWidgets.QLabel):
        """Title label that briefly shifts colour on an interval."""
        def __init__(self, text, parent=None):
            super().__init__(text, parent)
            self._glitch = False
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(3500)

        def _tick(self):
            self._glitch = True
            self.update()
            QtCore.QTimer.singleShot(80, self._restore)

        def _restore(self):
            self._glitch = False
            self.update()

        def paintEvent(self, event):
            if self._glitch:
                self.setStyleSheet(f"color: {CP_MAGENTA}; font-size: 22px; font-weight: 700;"
                                   f" letter-spacing: 6px; background: transparent;")
            else:
                self.setStyleSheet(f"color: {CP_CYAN}; font-size: 22px; font-weight: 700;"
                                   f" letter-spacing: 6px; background: transparent;")
            super().paintEvent(event)

    class SyncIndicator(QtWidgets.QWidget):
        """Animated blinking dot that shows sync is running."""
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFixedSize(10, 10)
            self._on = True
            t = QtCore.QTimer(self)
            t.timeout.connect(self._blink)
            t.start(600)

        def _blink(self):
            self._on = not self._on
            self.update()

        def paintEvent(self, event):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            colour = QtGui.QColor(CP_CYAN if self._on else CP_DIM)
            p.setBrush(colour)
            p.setPen(QtCore.Qt.NoPen)
            p.drawEllipse(1, 1, 8, 8)

    class WiZControlPanel(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("WiZ // CYBER SYNC")
            self.setMinimumSize(420, 370)
            self.setMaximumWidth(520)
            if os.path.exists(icon_path):
                self.setWindowIcon(QtGui.QIcon(icon_path))
            self.setStyleSheet(STYLESHEET)

            # ── Header bar ──────────────────────────────────────
            header = QtWidgets.QFrame()
            header.setObjectName("headerBar")
            header.setFixedHeight(64)
            h_layout = QtWidgets.QVBoxLayout(header)
            h_layout.setContentsMargins(20, 8, 20, 8)
            h_layout.setSpacing(1)

            title_lbl = GlitchLabel("WiZ SYNC")
            title_lbl.setObjectName("titleLabel")
            title_lbl.setAlignment(QtCore.Qt.AlignCenter)

            sub_lbl = QtWidgets.QLabel("AMBIENT LIGHT INTERFACE v2.0")
            sub_lbl.setObjectName("subtitleLabel")
            sub_lbl.setAlignment(QtCore.Qt.AlignCenter)

            h_layout.addWidget(title_lbl)
            h_layout.addWidget(sub_lbl)

            sep_cyan = QtWidgets.QFrame()
            sep_cyan.setObjectName("cyanLine")

            # ── Status row (sync indicator + IP) ────────────────
            status_row = QtWidgets.QHBoxLayout()
            status_row.setContentsMargins(20, 6, 20, 6)
            self._sync_dot = SyncIndicator()
            sync_tag = QtWidgets.QLabel("SYNC ACTIVE")
            sync_tag.setObjectName("sectionTag")
            self.device_label = QtWidgets.QLabel()
            self.device_label.setObjectName("deviceLabel")
            self.device_label.setAlignment(QtCore.Qt.AlignRight)
            status_row.addWidget(self._sync_dot)
            status_row.addSpacing(6)
            status_row.addWidget(sync_tag)
            status_row.addStretch(1)
            status_row.addWidget(self.device_label)

            sep_mag = QtWidgets.QFrame()
            sep_mag.setObjectName("magentaLine")

            # ── Power panel ──────────────────────────────────────
            power_panel = QtWidgets.QFrame()
            power_panel.setObjectName("panel")
            p_layout = QtWidgets.QVBoxLayout(power_panel)
            p_layout.setContentsMargins(16, 12, 16, 12)
            p_layout.setSpacing(8)

            pwr_tag = QtWidgets.QLabel("[ POWER CONTROL ]")
            pwr_tag.setObjectName("sectionTag")
            pwr_tag.setAlignment(QtCore.Qt.AlignCenter)

            self.power_button = QtWidgets.QPushButton("■  SYSTEM ON")
            self.power_button.setObjectName("powerBtn")
            self.power_button.setCheckable(True)
            self.power_button.setChecked(power_state)
            self.power_button.clicked.connect(self.toggle_power)

            p_layout.addWidget(pwr_tag)
            p_layout.addWidget(self.power_button)

            # ── Brightness panel ─────────────────────────────────
            bright_panel = QtWidgets.QFrame()
            bright_panel.setObjectName("panel")
            b_layout = QtWidgets.QVBoxLayout(bright_panel)
            b_layout.setContentsMargins(16, 12, 16, 14)
            b_layout.setSpacing(6)

            bright_tag = QtWidgets.QLabel("[ LUMINANCE OUTPUT ]")
            bright_tag.setObjectName("sectionTag")
            bright_tag.setAlignment(QtCore.Qt.AlignCenter)

            self.brightness_label = QtWidgets.QLabel()
            self.brightness_label.setObjectName("valueLabel")
            self.brightness_label.setAlignment(QtCore.Qt.AlignCenter)

            self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.brightness_slider.setRange(BRIGHTNESS_MIN, BRIGHTNESS_MAX)
            self.brightness_slider.setSingleStep(1)
            self.brightness_slider.setPageStep(5)
            self.brightness_slider.setValue(brightness_level)
            self.brightness_slider.valueChanged.connect(self.change_brightness)

            b_layout.addWidget(bright_tag)
            b_layout.addWidget(self.brightness_label)
            b_layout.addWidget(self.brightness_slider)

            # ── Vivid mode toggle ────────────────────────────────
            vivid_row = QtWidgets.QHBoxLayout()
            vivid_row.setContentsMargins(20, 4, 20, 4)
            vivid_tag = QtWidgets.QLabel("COLOUR MODE:")
            vivid_tag.setObjectName("sectionTag")
            self.vivid_button = QtWidgets.QPushButton("◈  VIVID")
            self.vivid_button.setObjectName("vividBtn")
            self.vivid_button.setCheckable(True)
            self.vivid_button.setChecked(use_vivid_mode)
            self.vivid_button.clicked.connect(self.toggle_vivid)
            vivid_row.addWidget(vivid_tag)
            vivid_row.addStretch(1)
            vivid_row.addWidget(self.vivid_button)

            sep_foot = QtWidgets.QFrame()
            sep_foot.setObjectName("cyanLine")

            # ── Footer ───────────────────────────────────────────
            footer_row = QtWidgets.QHBoxLayout()
            footer_row.setContentsMargins(20, 6, 20, 6)
            hint = QtWidgets.QLabel("SPC: power  |  V: vivid  |  ↑↓: brightness")
            hint.setObjectName("deviceLabel")
            footer_row.addWidget(hint)

            # ── Root layout ──────────────────────────────────────
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
            root.addWidget(header)
            root.addWidget(sep_cyan)
            root.addLayout(status_row)
            root.addWidget(sep_mag)
            root.addSpacing(10)
            root.addWidget(power_panel)
            root.addSpacing(8)
            root.addWidget(bright_panel)
            root.addSpacing(6)
            root.addLayout(vivid_row)
            root.addStretch(1)
            root.addWidget(sep_foot)
            root.addLayout(footer_row)

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh_state)
            self.timer.start(250)
            self.refresh_state()

        def toggle_power(self, checked: bool):
            set_power_state(checked)
            self.refresh_state()

        def toggle_vivid(self, checked: bool):
            global use_vivid_mode
            use_vivid_mode = checked
            print(f"[MODE] {'Vivid' if use_vivid_mode else 'Normal'}")
            self.refresh_state()

        def change_brightness(self, value: int):
            set_manual_brightness(value)
            self.refresh_state()

        def refresh_state(self):
            # Power button
            self.power_button.setChecked(power_state)
            self.power_button.setText("■  SYSTEM ON" if power_state else "□  SYSTEM OFF")

            # Brightness
            if self.brightness_slider.value() != brightness_level:
                self.brightness_slider.blockSignals(True)
                self.brightness_slider.setValue(brightness_level)
                self.brightness_slider.blockSignals(False)
            self.brightness_label.setText(f"{brightness_level}%")

            # Vivid
            self.vivid_button.setChecked(use_vivid_mode)
            self.vivid_button.setText("◈  VIVID" if use_vivid_mode else "◇  NORMAL")

            # Device
            self.device_label.setText(f"{IP_ADDRESS}:{PORT}")

            if not sync_thread.is_alive():
                QtWidgets.QApplication.quit()

        def closeEvent(self, event):
            global sync_active
            sync_active = False
            event.accept()

    window = WiZControlPanel()
    window.show()
    result = app.exec_()
    sync_active = False
    sync_thread.join(timeout=3.0)
    return result

# ------------------- MAIN -------------------
def configure_from_args(args: argparse.Namespace):
    global IP_ADDRESS, REFRESH_RATE_HZ, SLEEP_INTERVAL, ZONE_SIZE_PX, DEBUG_VIEW
    global use_vivid_mode, light_gamma, light_channel_gains, manual_brightness_control

    IP_ADDRESS = args.ip
    REFRESH_RATE_HZ = max(1, args.rate)
    SLEEP_INTERVAL = 1.0 / REFRESH_RATE_HZ
    ZONE_SIZE_PX = max(8, args.zone_size)
    DEBUG_VIEW = args.debug
    use_vivid_mode = not args.normal or args.vivid
    manual_brightness_control = not args.no_gui
    light_gamma = float(np.clip(args.light_gamma, 0.45, 1.60))
    light_channel_gains = np.array([
        np.clip(args.red_gain, 0.50, 1.50),
        np.clip(args.green_gain, 0.50, 1.50),
        np.clip(args.blue_gain, 0.50, 1.50),
    ], dtype=np.float32)
    logging.getLogger().setLevel(args.log_level)

def run_sync_loop(args: argparse.Namespace):
    global power_state, brightness_level, sync_active, last_color, use_vivid_mode, brightness_offset

    # Auto discover IP
    if args.no_discovery:
        print(f"[INFO] Discovery skipped. Using {IP_ADDRESS}:{PORT}")
    else:
        print("[INFO] Discovering WiZ Batten on local network...")
        discover_wiz_ip()

    print("[INFO] WiZ Batten Local Colour Sync Started")
    print(f"[INFO] Device -> {IP_ADDRESS}:{PORT}")
    if args.no_gui:
        print("[INFO] Controls: Up/Down brightness bias | Space power | V vivid | Ctrl+C quit")
    else:
        print("[INFO] Controls: GUI power + brightness | Space power | V vivid | close window to quit")

    if not args.no_startup_test and not ensure_device_ready():
        return

    monitor = get_screen_capture().monitors[0]
    screen_w, screen_h = monitor["width"], monitor["height"]

    try:
        while sync_active:
            loop_start = time.time()

            # Keyboard controls
            if key_pressed_once("up"):
                if manual_brightness_control:
                    set_manual_brightness(brightness_level + BRIGHTNESS_STEP)
                else:
                    brightness_offset = min(BRIGHTNESS_OFFSET_MAX, brightness_offset + BRIGHTNESS_STEP)
                    print(f"[BRIGHTNESS OFFSET] {brightness_offset:+d}")
            elif key_pressed_once("down"):
                if manual_brightness_control:
                    set_manual_brightness(brightness_level - BRIGHTNESS_STEP)
                else:
                    brightness_offset = max(BRIGHTNESS_OFFSET_MIN, brightness_offset - BRIGHTNESS_STEP)
                    print(f"[BRIGHTNESS OFFSET] {brightness_offset:+d}")
            elif key_pressed_once("space"):
                set_power_state(not power_state)
            elif key_pressed_once("v"):
                use_vivid_mode = not use_vivid_mode
                print(f"[MODE] {'Vivid' if use_vivid_mode else 'Normal'}")

            if not power_state:
                send_udp_command({"id": 1, "method": "setPilot", "params": {"state": False}})
                time.sleep(SLEEP_INTERVAL)
                continue

            # Capture screen zones
            zones_bgr = [capture_zone(rx, ry, ZONE_SIZE_PX, screen_w, screen_h)
                        for rx, ry, _ in ZONE_CFG]

            try:
                # Process color — isolated so a frame error never kills the loop
                target_rgb, auto_dim = process_zones(zones_bgr)
            except Exception as e:
                logging.warning("Frame processing error (skipping frame): %s", e)
                time.sleep(SLEEP_INTERVAL)
                continue

            if not manual_brightness_control:
                brightness_level = apply_brightness_offset(auto_dim)

            # Smooth transition
            smooth_rgb = interpolate(last_color, target_rgb)
            last_color = smooth_rgb
            print_status(smooth_rgb)

            draw_debug_view(zones_bgr, smooth_rgb)
            output_rgb = apply_light_calibration(smooth_rgb)

            # Send to light
            payload = {
                "id": 1,
                "method": "setPilot",
                "params": {
                    "state": True,
                    "r": int(output_rgb[0]),
                    "g": int(output_rgb[1]),
                    "b": int(output_rgb[2]),
                    "dimming": int(brightness_level),
                }
            }
            send_udp_command(payload)

            # Rate control
            elapsed = time.time() - loop_start
            time.sleep(max(0, SLEEP_INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        logging.error("Sync loop crashed: %s", e, exc_info=True)
    finally:
        if not args.keep_on_exit:
            send_udp_command({"id": 1, "method": "setPilot", "params": {"state": False}})
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    configure_from_args(args)

    if args.no_gui:
        run_sync_loop(args)
        return

    if importlib.util.find_spec("PyQt5") is None:
        print("[ERROR] PyQt5 is not installed. Install it with: pip install PyQt5")
        return

    worker = threading.Thread(target=run_sync_loop, args=(args,), daemon=True)
    worker.start()
    sys.exit(launch_gui(worker))

if __name__ == "__main__":
    main()
