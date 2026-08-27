"""Sensor dimension database for camera models and sensor formats."""

from typing import Dict, Optional, Tuple


# Known sensor physical dimensions in millimeters: (sensor_width_mm, sensor_height_mm)
CAMERA_SENSOR_DATABASE: Dict[str, Tuple[float, float]] = {
    # DJI Drones
    "dji fc330": (6.17, 4.55),       # Phantom 3 / 4 Standard (1/2.3")
    "dji fc6310": (13.2, 8.8),       # Phantom 4 Pro / RTK (1-inch)
    "dji fc220": (6.17, 4.55),       # Mavic Pro (1/2.3")
    "dji fc3411": (13.2, 8.8),      # Mavic 2 Pro (1-inch Hasselblad)
    "dji fc3582": (17.3, 13.0),     # Mavic 3 / 3 Enterprise (4/3 CMOS)
    "dji l1d-20c": (13.2, 8.8),     # Mavic 2 Pro Hasselblad
    "dji zenmuse p1": (35.9, 24.0),  # Full-frame survey camera
    "dji zenmuse h20": (7.68, 5.76), # 1/1.7" CMOS
    "dji mini 2": (6.17, 4.55),
    "dji mini 3 pro": (9.8, 7.3),    # 1/1.3" CMOS
    "dji mini 4 pro": (9.8, 7.3),

    # Popular Survey & Mirrorless / DSLR
    "sony ilce-7rm4": (35.7, 23.8),  # Sony A7R IV
    "sony ilce-7rm3": (35.9, 24.0),  # Sony A7R III
    "sony ilce-7m3": (35.6, 23.8),   # Sony A7 III
    "sony ilce-6000": (23.5, 15.6),  # Sony A6000 APS-C
    "canon eos 5d mark iv": (36.0, 24.0),
    "canon eos r5": (36.0, 24.0),
    "nikon d850": (35.9, 23.9),
    "nikon z7": (35.9, 23.9),

    # Action / Mobile
    "gopro hero10": (6.17, 4.55),
    "gopro hero11": (6.4, 5.6),
    "gopro hero12": (6.4, 5.6),
    "apple iphone 14 pro": (9.8, 7.3),
    "apple iphone 15 pro": (9.8, 7.3),
}

# Standard sensor format fallbacks
SENSOR_FORMAT_DIMENSIONS: Dict[str, Tuple[float, float]] = {
    "full_frame": (36.0, 24.0),
    "aps-c": (23.5, 15.6),
    "micro_four_thirds": (17.3, 13.0),
    "1_inch": (13.2, 8.8),
    "1/1.3_inch": (9.8, 7.3),
    "1/1.7_inch": (7.6, 5.7),
    "1/2.3_inch": (6.17, 4.55),
    "1/3_inch": (4.8, 3.6),
}


def lookup_sensor_dimensions(make: Optional[str], model: Optional[str]) -> Optional[Tuple[float, float]]:
    """Look up sensor physical dimensions in mm by camera make and model string."""
    if not model and not make:
        return None

    query = f"{make or ''} {model or ''}".strip().lower()

    # Exact key match
    if query in CAMERA_SENSOR_DATABASE:
        return CAMERA_SENSOR_DATABASE[query]

    # Partial substring match
    for key, dims in CAMERA_SENSOR_DATABASE.items():
        if key in query or (model and key in model.lower()):
            return dims

    return None
