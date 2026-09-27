#!/usr/bin/env python3
"""Independent normative FE identity byte vector; no ROS or production helper."""

import hashlib
import struct

payload = b"LIO-FE-OBS-V1\0" + bytes(range(16)) + struct.pack(">QqI", 7, -123456789, 6) + b"sensor"
assert hashlib.sha256(payload).hexdigest() == ("473a67e07bf10233a344f6f6728c053671149c3ec3cf5e747fbf4d2bfa875a16")
