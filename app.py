"""
================================================================================
 AUTONOMOUS DATA CENTER VISION INTELLIGENCE PLATFORM  (v1.1 - motion fix)
================================================================================
 Production-style multi-agent computer-vision system for real-time monitoring
 of data-center floors: technicians, equipment, racks, doors, safety hazards
 and restricted zones.

 QUICKSTART
 ----------
     pip install -r requirements.txt
     streamlit run app.py

 The app boots directly into LIVE DEMO MODE: a procedurally-rendered data-hall
 scene drives the entire agent pipeline (Vision -> Tracking -> Zone -> Event ->
 Risk -> Investigation -> Reporting), so the dashboard is never blank, even
 without a camera, GPU, or model weights.

 FIXES IN THIS VERSION
 ---------------------
   * Demo actors animate on WALL-CLOCK time (never stalls with loop timing).
   * Faster, clearly visible motion + walking bob + HUD frame counter.
   * Render loop is exception-safe (errors are shown, not silently frozen).
   * Tracker cleanup + file-upload path fixed.

 SAFETY & PRIVACY POLICY
 -----------------------
 Visual detections are OBSERVATIONS only. No identity or access-control
 decision is ever made from appearance alone; zone-entry events always request
 credential verification and human oversight.

 DATABASE (PostgreSQL-ready)
 ---------------------------
 Set env  DATABASE_URL=postgresql://user:pass@host:5432/dcvision  to use
 PostgreSQL; otherwise a local SQLite file is used. Schema auto-created.
================================================================================
"""

from __future__ import annotations

import os
import random
import tempfile
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Optional heavy dependencies (graceful fallbacks everywhere)
# --------------------------------------------------------------------------
try:
    from sqlalchemy import (
        Column, DateTime, Float, Integer, MetaData, String, Table, Text,
        create_engine,
    )
    SQLALCHEMY_OK = True
except Exception:
    SQLALCHEMY_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception:
    YOLO_OK = False

try:
    import av  # PyAV fallback decoder
    AV_OK = True
except Exception:
    AV_OK = False

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
CONFIG = {
    "app_title": "Autonomous Data Center Vision Intelligence Platform",
    "org": "DC-OPS // Facility Intelligence",
    "default_camera": "CAM-01",
    "database_url": os.environ.get("DATABASE_URL", "sqlite:///datacenter_vision.db"),
    "demo_fps": 24,
    "demo_size": (1280, 720),
    "timeline_max": 600,
    "active_event_window_s": 45,
    "zones": [
        {"id": "server_room", "label": "SERVER ROOM", "color": (200, 160, 40),
         "poly": [(0.04, 0.06), (0.44, 0.06), (0.44, 0.52), (0.04, 0.52)]},
        {"id": "restricted_area", "label": "RESTRICTED AREA", "color": (60, 60, 230),
         "poly": [(0.56, 0.06), (0.96, 0.06), (0.96, 0.42), (0.56, 0.42)]},
        {"id": "maintenance_area", "label": "MAINTENANCE", "color": (40, 180, 240),
         "poly": [(0.56, 0.56), (0.96, 0.56), (0.96, 0.94), (0.56, 0.94)]},
        {"id": "emergency_area", "label": "EMERGENCY", "color": (80, 80, 250),
         "poly": [(0.04, 0.60), (0.28, 0.60), (0.28, 0.94), (0.04, 0.94)]},
        {"id": "equipment_corridor", "label": "EQUIPMENT CORRIDOR", "color": (180, 120, 220),
         "poly": [(0.32, 0.56), (0.52, 0.56), (0.52, 0.96), (0.32, 0.96)]},
    ],
    "thresholds": {
        "restricted_dwell_s": 10.0,
        "door_open_s": 10.0,
        "unattended_s": 8.0,
        "unattended_radius": 150.0,
        "crowd_count": 4,
        "crowd_persist_s": 2.5,
        "abnormal_speed": 300.0,   # px/s
    },
    "cooldown_s": {
        "unauthorized_zone_entry": 20, "extended_presence": 25, "door_left_open": 20,
        "unattended_equipment": 25, "abnormal_movement": 15, "crowding": 25,
        "safety_hazard": 20,
    },
    "risk_base": {
        "unauthorized_zone_entry": 72, "extended_presence": 82, "door_left_open": 55,
        "unattended_equipment": 58, "abnormal_movement": 52, "crowding": 63,
        "safety_hazard": 76,
    },
}

EVENT_LABELS = {
    "unauthorized_zone_entry": "Unauthorized Zone Entry",
    "extended_presence": "Extended Restricted Presence",
    "door_left_open": "Door Left Open",
    "unattended_equipment": "Unattended Equipment",
    "abnormal_movement": "Abnormal Movement",
    "crowding": "Crowding",
    "safety_hazard": "Safety Hazard",
}

CLASS_COLORS = {
    "technician": (230, 190, 60), "person": (230, 190, 60),
    "toolbox": (60, 140, 255), "equipment": (200, 120, 255),
    "cart": (220, 120, 220), "door": (80, 220, 255), "unknown": (160, 160, 160),
}

SEV_COLOR = {
    "low": ("#14532d", "#86efac"), "medium": ("#713f12", "#fde047"),
    "high": ("#7c2d12", "#fdba74"), "critical": ("#7f1d1d", "#fca5a5"),
}

INVESTIGATION_PLAYBOOK = {
    "unauthorized_zone_entry": (
        "Unverified individual observed inside {zone}. Visual appearance cannot establish "
        "authorization; treat as observation pending credential check.",
        "Dispatch floor operator to verify badge, cross-check access-control logs for the "
        "affected door, and confirm escort policy compliance."),
    "extended_presence": (
        "Track persisted in {zone} beyond the permitted dwell window. Possible unscoped work "
        "or stalled activity.",
        "Request work-order correlation; if no matching ticket, ask on-duty supervisor to "
        "confirm activity and duration."),
    "door_left_open": (
        "Door associated with {zone} has remained open past the security threshold, weakening "
        "the physical access boundary.",
        "Verify door sensor state, dispatch nearest technician to close, and audit recent "
        "badge events at that portal."),
    "unattended_equipment": (
        "Equipment/asset stationary in {zone} with no personnel in proximity. Possible "
        "forgotten tooling or unlogged asset staging.",
        "Send operator to claim or tag the asset; check maintenance logs for scheduled "
        "staging before escalating."),
    "abnormal_movement": (
        "High-velocity movement detected in {zone}; inconsistent with normal aisle transit.",
        "Review the last 30 s of footage for the track, verify no emergency in progress, "
        "and remind crew of floor speed policy."),
    "crowding": (
        "Personnel density in {zone} exceeds the operational threshold, raising congestion "
        "and egress risk.",
        "Ask shift lead to redistribute personnel; verify no active incident assembly is "
        "taking place."),
    "safety_hazard": (
        "PPE / safety-rule violation observed in {zone}. Detections are observations and "
        "require human confirmation.",
        "Alert the safety officer, confirm PPE compliance on site, and log a corrective "
        "action entry."),
}


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class Detection:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    attrs: Dict = field(default_factory=dict)


@dataclass
class VisionEvent:
    event_id: str
    ts: float
    camera: str
    event_type: str
    track_id: Optional[int]
    object_class: str
    zone: str
    confidence: float
    details: str
    severity: str = "medium"
    risk_score: float = 0.0


class EventBus:
    def __init__(self):
        self._q: Dict[str, deque] = {}

    def publish(self, topic: str, msg):
        self._q.setdefault(topic, deque()).append(msg)

    def drain(self, topic: str) -> List:
        q = self._q.get(topic)
        if not q:
            return []
        items = list(q)
        q.clear()
        return items


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) +
          max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------
# SYNTHETIC DATA-CENTER SCENE  (wall-clock animated — never stalls)
# --------------------------------------------------------------------------
SCENARIOS = [
    ("NORMAL OPERATIONS", 16),
    ("RESTRICTED-AREA BREACH", 16),
    ("DOOR LEFT OPEN", 14),
    ("UNATTENDED EQUIPMENT", 16),
    ("CORRIDOR CROWDING", 14),
    ("ABNORMAL MOVEMENT + PPE FAULT", 16),
]


class SyntheticScene:
    FLOOR = (34, 28, 22)
    GRID = (48, 40, 32)

    def __init__(self, w=1280, h=720, fps=24, camera="CAM-01", seed=7):
        self.W, self.H, self.fps, self.camera = w, h, fps, camera
        self.rng = random.Random(seed)
        self.frame_idx = 0
        self.t = 0.0
        self.scenario_idx = 0
        self.scenario_t = 0.0
        self.actors: List[Dict] = []
        self._actor_seq = 0
        self.speed_mult = 1.0                      # user-adjustable demo speed
        self._last = time.time()                   # WALL-CLOCK animation base
        self.doors = {
            "R1": {"label": "R1 // RESTRICTED", "rel": (0.70, 0.78, 0.42), "open": False,
                   "open_for": 0.0, "open_target": 2.0},
            "S1": {"label": "S1 // SERVER", "rel": (0.28, 0.36, 0.52), "open": False,
                   "open_for": 0.0, "open_target": 2.0},
        }
        self._build_racks()
        self._noise = np.random.randint(0, 9, (h, w), dtype=np.uint8)
        self._spawn_scenario(0)

    # ---------------- setup ----------------
    def _build_racks(self):
        self.racks = []

        def grid(rx1, ry1, rx2, ry2, rows, cols):
            x1, y1 = rx1 * self.W, ry1 * self.H
            x2, y2 = rx2 * self.W, ry2 * self.H
            cw, ch = (x2 - x1) / cols, (y2 - y1) / rows
            for r in range(rows):
                for c in range(cols):
                    self.racks.append((int(x1 + c * cw + cw * 0.12),
                                       int(y1 + r * ch + ch * 0.14),
                                       int(cw * 0.76), int(ch * 0.72)))

        grid(0.06, 0.12, 0.42, 0.48, 2, 4)
        grid(0.58, 0.10, 0.94, 0.36, 1, 3)
        grid(0.58, 0.62, 0.94, 0.88, 1, 3)

    def _add_actor(self, kind, rx, ry, speed, home=None, wps=None,
                   authorized=True, helmet=True):
        self._actor_seq += 1
        self.actors.append({
            "id": self._actor_seq, "kind": kind,
            "x": rx * self.W, "y": ry * self.H,
            "speed": speed, "home": home, "wps": wps or [], "wpi": 0,
            "authorized": authorized, "helmet": helmet,
            "phase": self.rng.uniform(0, 6.28), "moving": False,
        })

    def _wander(self, rx1, ry1, rx2, ry2):
        return (rx1 * self.W + self.rng.random() * (rx2 - rx1) * self.W,
                ry1 * self.H + self.rng.random() * (ry2 - ry1) * self.H)

    def _spawn_scenario(self, idx):
        self.actors = []
        self._actor_seq = 0
        for d in self.doors.values():
            d["open"], d["open_for"] = False, 0.0
        srv = (0.06, 0.14, 0.42, 0.48)
        cor = (0.33, 0.58, 0.51, 0.94)
        mnt = (0.58, 0.60, 0.94, 0.92)

        if idx == 0:    # normal ops — busy floor
            self._add_actor("technician", 0.15, 0.30, 115, home=srv)
            self._add_actor("technician", 0.40, 0.70, 120, home=cor)
            self._add_actor("technician", 0.30, 0.42, 105, home=srv)
            self._add_actor("cart", 0.42, 0.85, 90, home=cor)
            self._add_actor("toolbox", 0.20, 0.40, 34, home=srv)
        elif idx == 1:  # restricted breach
            self._add_actor("technician", 0.15, 0.25, 110, home=srv)
            self._add_actor("technician", 0.60, 0.50, 125, authorized=False,
                            wps=[(0.64 * self.W, 0.38 * self.H),
                                 (0.78 * self.W, 0.20 * self.H)])
        elif idx == 2:  # door left open
            self.doors["R1"]["open"] = True
            self._add_actor("technician", 0.66, 0.50, 100,
                            home=(0.58, 0.44, 0.92, 0.54))
            self._add_actor("cart", 0.80, 0.50, 85,
                            home=(0.58, 0.44, 0.92, 0.54))
        elif idx == 3:  # unattended toolbox
            self._add_actor("toolbox", 0.42, 0.76, 0.0)
            a = self._add_actor("technician", 0.43, 0.74, 115,
                                wps=[(0.36 * self.W, 0.60 * self.H),
                                     (0.18 * self.W, 0.30 * self.H)]) or None
            self.actors[-1]["home"] = srv
            self._add_actor("technician", 0.10, 0.20, 105, home=srv)
        elif idx == 4:  # crowding
            for _ in range(5):
                self._add_actor("technician",
                                0.34 + self.rng.random() * 0.15,
                                0.62 + self.rng.random() * 0.28,
                                70, home=cor)
        elif idx == 5:  # runner + PPE fault
            self._add_actor("technician", 0.34, 0.92, 380,
                            wps=[(0.34 * self.W, 0.60 * self.H),
                                 (0.50 * self.W, 0.60 * self.H),
                                 (0.50 * self.W, 0.93 * self.H),
                                 (0.34 * self.W, 0.93 * self.H)])
            self._add_actor("technician", 0.70, 0.75, 95, home=mnt, helmet=False)

    # ---------------- step (wall-clock dt) ----------------
    def step(self):
        now = time.time()
        dt = clamp(now - self._last, 0.0, 0.10) * max(0.0, self.speed_mult)
        self._last = now
        if dt <= 0:
            dt = 1e-4

        self.t += dt
        self.frame_idx += 1
        self.scenario_t += dt
        if self.scenario_t >= SCENARIOS[self.scenario_idx][1]:
            self.scenario_idx = (self.scenario_idx + 1) % len(SCENARIOS)
            self.scenario_t = 0.0
            self._spawn_scenario(self.scenario_idx)

        # doors lifecycle
        for d in self.doors.values():
            if d["open"]:
                d["open_for"] += dt
                if self.scenario_idx != 2 and d["open_for"] > d["open_target"]:
                    d["open"], d["open_for"] = False, 0.0
            elif self.rng.random() < 0.0025:
                d["open"] = True
                d["open_target"] = 2.0 + self.rng.random() * 2.5

        # ---- actor movement (this is what animates technicians/equipment) ----
        for a in self.actors:
            target = None
            if a["wps"]:
                tx, ty = a["wps"][a["wpi"]]
                if (a["x"] - tx) ** 2 + (a["y"] - ty) ** 2 < 100:
                    if a["home"]:
                        a["wps"], a["wpi"] = [], 0
                    else:
                        a["wpi"] = (a["wpi"] + 1) % len(a["wps"])
                else:
                    target = (tx, ty)
            if target is None and a["home"]:
                wp = a.get("_wp")
                if not wp or (a["x"] - wp[0]) ** 2 + (a["y"] - wp[1]) ** 2 < 100:
                    a["_wp"] = self._wander(*a["home"])
                target = a["_wp"]

            if target and a["speed"] > 0:
                dx, dy = target[0] - a["x"], target[1] - a["y"]
                dist = max(1e-3, (dx * dx + dy * dy) ** 0.5)
                step = min(dist, a["speed"] * dt)
                a["x"] += dx / dist * step
                a["y"] += dy / dist * step
                a["moving"] = step > 0.2
            else:
                a["moving"] = False

        return self._render(), self._detections(), self.doors, \
            SCENARIOS[self.scenario_idx][0]

    def _detections(self):
        sizes = {"technician": (34, 86), "toolbox": (40, 26), "cart": (66, 46)}
        out = []
        for a in self.actors:
            w, h = sizes[a["kind"]]
            out.append(Detection(
                cls=a["kind"], x1=a["x"] - w / 2, y1=a["y"] - h,
                x2=a["x"] + w / 2, y2=a["y"],
                conf=round(min(0.98, 0.80 + self.rng.uniform(0.0, 0.18)), 2),
                attrs={"authorized": a["authorized"], "helmet": a["helmet"]}))
        return out

    # ---------------- rendering ----------------
    def _render(self):
        W, H = self.W, self.H
        img = np.full((H, W, 3), self.FLOOR, dtype=np.uint8)
        for x in range(0, W, 64):
            cv2.line(img, (x, 0), (x, H), self.GRID, 1)
        for y in range(0, H, 64):
            cv2.line(img, (0, y), (W, y), self.GRID, 1)

        for i, (x, y, w, h) in enumerate(self.racks):
            cv2.rectangle(img, (x, y), (x + w, y + h), (72, 62, 50), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (110, 96, 78), 1)
            for r in range(3):
                for c in range(6):
                    lx = x + 6 + c * (w - 12) // 6
                    ly = y + 8 + r * (h - 16) // 3
                    blink = (self.frame_idx // 8 + i * 3 + r * 5 + c * 7) % 9 == 0
                    col = (60, 170, 255) if blink else (90, 200, 90)
                    if (i + r + c) % 11 == 0:
                        col = (60, 160, 240)
                    cv2.circle(img, (lx, ly), 1, col, -1)

        for key, d in self.doors.items():
            rx1, rx2, ry = d["rel"]
            x1, x2, y = int(rx1 * W), int(rx2 * W), int(ry * H)
            cv2.rectangle(img, (x1 - 4, y - 6), (x2 + 4, y + 6), (120, 110, 90), 2)
            if d["open"]:
                warn = d["open_for"] > CONFIG["thresholds"]["door_open_s"]
                pulse = 0.5 + 0.5 * np.sin(self.t * 6) if warn else 0.0
                col = (int(60 + 120 * pulse), 70, int(220 - 160 * pulse)) if warn else (120, 210, 90)
                cv2.rectangle(img, (x1, y - 4), (x1 + int((x2 - x1) * 0.3), y + 4), col, -1)
                cv2.putText(img, f"{key} OPEN {int(d['open_for'])}s", (x1, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (x1, y - 4), (x2, y + 4), (150, 140, 120), -1)

        for a in self.actors:
            self._draw_actor(img, a)

        # moving noise shimmer — extra proof the feed is live
        roll = (self.frame_idx * 7) % 64
        noise = np.roll(self._noise, roll, axis=1)
        img = cv2.add(img, cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR) // 4)
        return img

    def _draw_actor(self, img, a):
        x, y = int(a["x"]), int(a["y"])
        bob = int(np.sin(self.t * 9 + a["phase"]) * 2.5) if a["moving"] else 0
        cv2.ellipse(img, (x, y), (16, 6), 0, 0, 360, (20, 16, 12), -1)
        if a["kind"] == "technician":
            yb = y + bob
            cv2.rectangle(img, (x - 13, yb - 56), (x + 13, yb - 4), (200, 130, 50), -1)
            cv2.rectangle(img, (x - 13, yb - 56), (x + 13, yb - 4), (240, 170, 80), 1)
            cv2.line(img, (x, yb - 54), (x, yb - 6), (230, 160, 70), 1)
            if a["helmet"]:
                cv2.circle(img, (x, yb - 64), 9, (60, 200, 255), -1)
                cv2.rectangle(img, (x - 11, yb - 64), (x + 11, yb - 61), (60, 200, 255), -1)
            else:
                cv2.circle(img, (x, yb - 64), 8, (120, 100, 170), -1)
        elif a["kind"] == "toolbox":
            cv2.rectangle(img, (x - 18, y - 22), (x + 18, y), (40, 120, 230), -1)
            cv2.rectangle(img, (x - 18, y - 22), (x + 18, y), (80, 160, 255), 1)
            cv2.rectangle(img, (x - 6, y - 27), (x + 6, y - 22), (80, 160, 255), 2)
        elif a["kind"] == "cart":
            cv2.rectangle(img, (x - 30, y - 42), (x + 30, y - 8), (170, 110, 190), -1)
            cv2.rectangle(img, (x - 30, y - 42), (x + 30, y - 8), (210, 150, 230), 1)
            spin = int(self.t * 12) % 2 == 0
            for wx in (x - 22, x + 22):
                cv2.circle(img, (wx, y - 5), 5, (120, 120, 120) if spin else (90, 90, 90), -1)


# --------------------------------------------------------------------------
# DETECTION BACKENDS
# --------------------------------------------------------------------------
class YoloDetector:
    MAP = {0: "technician", 24: "toolbox", 26: "toolbox", 28: "equipment",
           2: "cart", 7: "cart"}

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.status = "YOLOv8n (Ultralytics) ready"

    def detect(self, frame) -> List[Detection]:
        try:
            res = self.model.predict(frame, verbose=False, conf=0.35)[0]
        except Exception:
            return []
        out = []
        for b in res.boxes:
            c = int(b.cls.item())
            if c not in self.MAP:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append(Detection(self.MAP[c], x1, y1, x2, y2, float(b.conf.item()), {}))
        return out


class AVFileSource:
    def __init__(self, path):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self._gen = self._decode()

    def _decode(self):
        while True:
            self.container.seek(0)
            for frame in self.container.decode(self.stream):
                yield frame.to_ndarray(format="bgr24")

    def read(self):
        try:
            return True, next(self._gen)
        except Exception:
            return False, None


# --------------------------------------------------------------------------
# AGENTS
# --------------------------------------------------------------------------
class VisionAgent:
    name = "Vision Agent"

    def __init__(self, source_type: str, camera: str, path: str = "", url: str = ""):
        self.source_type, self.camera = source_type, camera
        self.scene = None
        self.cap = None
        self.av_src = None
        self.detector = None
        self.detector_status = "synthetic ground-truth (demo)"
        self.fps = CONFIG["demo_fps"]

        if source_type == "demo":
            self.scene = SyntheticScene(CONFIG["demo_size"][0], CONFIG["demo_size"][1],
                                        CONFIG["demo_fps"], camera)
        elif source_type == "webcam":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Webcam unavailable — falling back to demo mode.")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self._init_yolo()
        elif source_type == "file":
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                if AV_OK:
                    self.av_src = AVFileSource(path)
                    self.cap = None
                else:
                    raise RuntimeError("Could not open uploaded video.")
            self.fps = (self.cap.get(cv2.CAP_PROP_FPS) or 30) if self.cap else 30
            self._init_yolo()
        elif source_type == "rtsp":
            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                raise RuntimeError("RTSP stream unreachable — falling back to demo mode.")
            self.fps = 25
            self._init_yolo()

    def _init_yolo(self):
        if YOLO_OK:
            try:
                self.detector = YoloDetector()
                self.detector_status = self.detector.status
            except Exception as e:
                self.detector_status = f"YOLO unavailable ({type(e).__name__}) — raw feed only"
        else:
            self.detector_status = "Ultralytics not installed — raw feed only"

    def read(self):
        if self.scene is not None:
            return self.scene.step()
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                if self.source_type == "file":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                if not ok:
                    return None, [], None, "SIGNAL LOST"
        else:
            ok, frame = self.av_src.read()
            if not ok:
                return None, [], None, "SIGNAL LOST"
        dets = self.detector.detect(frame) if self.detector else []
        return frame, dets, None, "LIVE FEED"

    def release(self):
        if self.cap is not None:
            self.cap.release()


class Track:
    __slots__ = ("id", "cls", "bbox", "conf", "attrs", "vx", "vy", "speed",
                 "hits", "miss", "zone", "prev_zone", "zone_dwell",
                 "still_time", "first_seen")

    def __init__(self, tid, det: Detection):
        self.id, self.cls = tid, det.cls
        self.bbox = (det.x1, det.y1, det.x2, det.y2)
        self.conf, self.attrs = det.conf, dict(det.attrs)
        self.vx = self.vy = self.speed = 0.0
        self.hits, self.miss = 1, 0
        self.zone = self.prev_zone = None
        self.zone_dwell = self.still_time = 0.0
        self.first_seen = time.time()

    @property
    def center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def foot(self):
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])


class TrackingAgent:
    """ByteTrack-style IoU tracker with motion extrapolation (BoT-SORT-ready)."""

    name = "Tracking Agent"

    def __init__(self, max_age=14, min_iou=0.22):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.max_age, self.min_iou = max_age, min_iou
        self.total_ids = 0

    def update(self, dets: List[Detection], dt: float) -> List[Track]:
        # 1) motion prediction
        for tr in self.tracks:
            cx, cy = tr.center
            nx, ny = cx + tr.vx * dt, cy + tr.vy * dt
            w = tr.bbox[2] - tr.bbox[0]
            h = tr.bbox[3] - tr.bbox[1]
            tr.bbox = (nx - w / 2, ny - h / 2, nx + w / 2, ny + h / 2)

        # 2) greedy IoU matching
        pairs = []
        for di, d in enumerate(dets):
            for ti, tr in enumerate(self.tracks):
                v = iou((d.x1, d.y1, d.x2, d.y2), tr.bbox)
                if v >= self.min_iou:
                    pairs.append((v, di, ti))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for v, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            tr, d = self.tracks[ti], dets[di]
            prev = tr.center
            tr.bbox = (d.x1, d.y1, d.x2, d.y2)
            tr.conf, tr.attrs, tr.hits, tr.miss = d.conf, dict(d.attrs), tr.hits + 1, 0
            cx, cy = tr.center
            if dt > 0:
                tr.vx = 0.6 * ((cx - prev[0]) / dt) + 0.4 * tr.vx
                tr.vy = 0.6 * ((cy - prev[1]) / dt) + 0.4 * tr.vy
            tr.speed = (tr.vx ** 2 + tr.vy ** 2) ** 0.5
            tr.still_time = tr.still_time + dt if tr.speed < 10 else 0.0

        # 3) unmatched detections -> new tracks
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(self.next_id, d))
                self.next_id += 1
                self.total_ids += 1

        # 4) unmatched tracks -> miss, then prune
        for ti, tr in enumerate(self.tracks):
            if ti < len(used_t | set()) and ti not in used_t and ti < self.next_id:
                if ti >= 0 and ti not in used_t and self.tracks[ti].miss >= 0:
                    if ti not in used_d:  # index domains differ; guard below
                        pass
        for ti in range(len(self.tracks)):
            matched_this = ti in used_t
            if not matched_this and ti < len([t for t in self.tracks]) and \
                    self.tracks[ti].hits > 0 and ti not in used_t:
                if ti < len(used_t) or True:
                    if ti not in used_t:
                        self.tracks[ti].miss += 1
        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]
        return self.tracks


class ZoneAgent:
    name = "Zone Agent"

    def __init__(self):
        self.polys = {}
        self.w = 0

    def configure(self, w, h):
        self.w, self.h = w, h
        self.polys = {}
        for z in CONFIG["zones"]:
            self.polys[z["id"]] = np.array(
                [(int(px * w), int(py * h)) for px, py in z["poly"]], dtype=np.int32)

    def zone_of(self, pt) -> Optional[str]:
        for zid, poly in self.polys.items():
            if cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0:
                return zid
        return None

    def update(self, tracks: List[Track], dt: float):
        for tr in tracks:
            z = self.zone_of(tr.foot)
            if z == tr.zone:
                tr.zone_dwell += dt
            else:
                tr.prev_zone, tr.zone, tr.zone_dwell = tr.zone, z, 0.0


class EventAgent:
    name = "Event Agent"

    def __init__(self, camera):
        self.camera = camera
        self.last_emit: Dict[str, float] = {}
        self.crowd_since: Dict[str, float] = {}
        self.emitted = 0

    def _emit(self, events, etype, now, zone="", conf=0.9, details="",
              oclass="technician", track_id=None):
        key = f"{etype}:{track_id if track_id is not None else zone}"
        if now - self.last_emit.get(key, -1e9) < CONFIG["cooldown_s"].get(etype, 15):
            return
        self.last_emit[key] = now
        self.emitted += 1
        events.append(VisionEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}", ts=now, camera=self.camera,
            event_type=etype, track_id=track_id, object_class=oclass, zone=zone,
            confidence=round(conf, 2), details=details))

    def update(self, tracks, doors, now, dt):
        T = CONFIG["thresholds"]
        ev = []
        techs = [t for t in tracks if t.cls in ("technician", "person")]
        zone_counts: Dict[str, int] = {}
        for tr in techs:
            if tr.zone:
                zone_counts[tr.zone] = zone_counts.get(tr.zone, 0) + 1

            if tr.zone == "restricted_area":
                if tr.prev_zone != "restricted_area" and tr.zone_dwell < dt * 2:
                    auth = tr.attrs.get("authorized")
                    if auth is False:
                        note = "Credential NOT verified (simulation metadata) — human confirmation required."
                    elif auth is True:
                        note = "Credential flagged valid in demo metadata — verify against access system."
                    else:
                        note = "Credential state unknown — verification required (no identity inference from appearance)."
                    self._emit(ev, "unauthorized_zone_entry", now, tr.zone, tr.conf,
                               f"{tr.cls.title()} #{tr.id} entered restricted area. {note}",
                               tr.cls, tr.id)
                if tr.zone_dwell > T["restricted_dwell_s"]:
                    self._emit(ev, "extended_presence", now, tr.zone, tr.conf,
                               f"Track #{tr.id} inside restricted area for {tr.zone_dwell:.0f}s "
                               f"(limit {T['restricted_dwell_s']:.0f}s).", tr.cls, tr.id)

            if tr.speed > T["abnormal_speed"]:
                self._emit(ev, "abnormal_movement", now, tr.zone or "floor", tr.conf,
                           f"Track #{tr.id} moving at {tr.speed:.0f} px/s (rapid transit).",
                           tr.cls, tr.id)

            if tr.zone == "maintenance_area" and tr.attrs.get("helmet") is False:
                self._emit(ev, "safety_hazard", now, tr.zone, tr.conf,
                           f"Track #{tr.id} in maintenance area without detected helmet (PPE check required).",
                           tr.cls, tr.id)
            if tr.zone == "emergency_area" and tr.zone_dwell > 1.0:
                self._emit(ev, "safety_hazard", now, tr.zone, tr.conf,
                           f"Track #{tr.id} present in emergency staging area.", tr.cls, tr.id)

        for zid, cnt in zone_counts.items():
            if cnt >= T["crowd_count"]:
                self.crowd_since.setdefault(zid, now)
                if now - self.crowd_since[zid] > T["crowd_persist_s"]:
                    self._emit(ev, "crowding", now, zid, 0.85,
                               f"{cnt} personnel detected in {zid} (threshold {T['crowd_count']}).",
                               "group", None)
            else:
                self.crowd_since.pop(zid, None)

        for tr in tracks:
            if tr.cls in ("toolbox", "cart", "equipment") and tr.still_time > T["unattended_s"]:
                near = any(((t.center[0] - tr.center[0]) ** 2 +
                            (t.center[1] - tr.center[1]) ** 2) ** 0.5 < T["unattended_radius"]
                           for t in techs)
                if not near:
                    self._emit(ev, "unattended_equipment", now, tr.zone or "floor", tr.conf,
                               f"{tr.cls.title()} #{tr.id} stationary for {tr.still_time:.0f}s "
                               f"with no personnel within {T['unattended_radius']:.0f}px.",
                               tr.cls, tr.id)

        if doors:
            for key, d in doors.items():
                if d["open"] and d["open_for"] > T["door_open_s"]:
                    self._emit(ev, "door_left_open", now,
                               "restricted_area" if key == "R1" else "server_room", 0.95,
                               f"Door {d['label']} open for {d['open_for']:.0f}s "
                               f"(limit {T['door_open_s']:.0f}s).", "door", None)
        return ev


class RiskAgent:
    name = "Risk Agent"

    def process(self, events: List[VisionEvent]) -> List[VisionEvent]:
        for e in events:
            base = CONFIG["risk_base"].get(e.event_type, 50)
            score = base * (0.6 + 0.4 * e.confidence)
            if e.event_type == "unauthorized_zone_entry" and "NOT verified" in e.details:
                score += 10
            if e.event_type == "extended_presence":
                score += 6
            e.risk_score = round(clamp(score, 5, 98), 1)
            e.severity = ("critical" if e.risk_score >= 80 else
                          "high" if e.risk_score >= 62 else
                          "medium" if e.risk_score >= 42 else "low")
        return events


class InvestigationAgent:
    name = "Investigation Agent"

    def __init__(self):
        self.analyses = deque(maxlen=80)
        self.processed = 0
        self.last_active = 0.0

    def process(self, events: List[VisionEvent]):
        now = time.time()
        for e in events:
            assess, resp = INVESTIGATION_PLAYBOOK.get(
                e.event_type,
                ("Unclassified observation requiring human review.",
                 "Route to on-duty operator."))
            self.analyses.appendleft({
                "id": f"inv_{uuid.uuid4().hex[:6]}", "ts": now,
                "event": EVENT_LABELS.get(e.event_type, e.event_type),
                "event_type": e.event_type, "severity": e.severity,
                "confidence": e.confidence, "risk": e.risk_score,
                "zone": e.zone or "—", "track": e.track_id,
                "assessment": assess.format(zone=e.zone or "the monitored floor"),
                "response": resp, "status": "ROUTED TO OPERATOR",
            })
            self.processed += 1
            self.last_active = now

    @property
    def status(self):
        return "ANALYZING" if time.time() - self.last_active < 3 else "IDLE"


class ReportingAgent:
    name = "Reporting Agent"

    def __init__(self):
        self.engine = None
        self.table = None
        self.rows = 0
        self.mode = "in-memory"
        if SQLALCHEMY_OK:
            try:
                url = CONFIG["database_url"]
                kw = {"future": True}
                if url.startswith("sqlite"):
                    kw["connect_args"] = {"check_same_thread": False}
                self.engine = create_engine(url, **kw)
                meta = MetaData()
                self.table = Table(
                    "vision_events", meta,
                    Column("id", Integer, primary_key=True),
                    Column("ts", DateTime(timezone=True), index=True),
                    Column("camera", String(64)),
                    Column("track_id", Integer, nullable=True),
                    Column("object_class", String(64)),
                    Column("zone", String(64)),
                    Column("event_type", String(64), index=True),
                    Column("severity", String(16)),
                    Column("confidence", Float),
                    Column("risk_score", Float),
                    Column("details", Text),
                )
                meta.create_all(self.engine)
                self.mode = "PostgreSQL" if url.startswith("postgresql") else "SQLite"
            except Exception as e:
                self.mode = f"in-memory (DB error: {type(e).__name__})"

    def persist(self, events: List[VisionEvent]):
        for e in events:
            self.rows += 1
            if self.engine is None or self.table is None:
                continue
            try:
                with self.engine.begin() as conn:
                    conn.execute(self.table.insert().values(
                        ts=datetime.fromtimestamp(e.ts, tz=timezone.utc),
                        camera=e.camera, track_id=e.track_id,
                        object_class=e.object_class, zone=e.zone or "",
                        event_type=e.event_type, severity=e.severity,
                        confidence=e.confidence, risk_score=e.risk_score,
                        details=e.details))
            except Exception:
                self.mode = "in-memory (write error)"


# --------------------------------------------------------------------------
# PIPELINE
# --------------------------------------------------------------------------
class Pipeline:
    def __init__(self, source_type="demo", camera=None, path="", url=""):
        self.camera = camera or CONFIG["default_camera"]
        self.vision = VisionAgent(source_type, self.camera, path, url)
        self.bus = EventBus()
        self.tracker = TrackingAgent()
        self.zones = ZoneAgent()
        self.events_agent = EventAgent(self.camera)
        self.risk = RiskAgent()
        self.investigator = InvestigationAgent()
        self.reporter = ReportingAgent()

        self.show_zones = True
        self.show_tracks = True
        self.show_hud = True

        self.timeline = deque(maxlen=CONFIG["timeline_max"])
        self.recent_events = deque(maxlen=200)
        self.risk_history = deque(maxlen=240)
        self.start_time = time.time()
        self._prev = time.time()
        self.fps_ema = 0.0
        self._last_hist = 0.0
        self.frames = 0

    def step(self):
        now = time.time()
        dt = clamp(now - self._prev, 1 / 60, 1 / 10)
        self._prev = now
        inst = 1 / dt if dt > 0 else 0
        self.fps_ema = 0.9 * self.fps_ema + 0.1 * inst if self.fps_ema else inst

        frame, dets, doors, scenario = self.vision.read()
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        H, W = frame.shape[:2]
        if not self.zones.polys or self.zones.w != W:
            self.zones.configure(W, H)

        self.bus.publish("detections", dets)
        tracks = self.tracker.update(self.bus.drain("detections")[0], dt)
        self.zones.update(tracks, dt)

        for e in self.events_agent.update(tracks, doors, now, dt):
            self.bus.publish("events", e)

        scored = self.risk.process(self.bus.drain("events"))
        self.investigator.process(scored)
        self.reporter.persist(scored)

        for e in scored:
            self.recent_events.append(e)
            self.timeline.appendleft({
                "Time": datetime.fromtimestamp(e.ts).strftime("%H:%M:%S"),
                "Camera": e.camera, "Event": EVENT_LABELS.get(e.event_type, e.event_type),
                "Zone": e.zone or "—", "Class": e.object_class,
                "Track": e.track_id if e.track_id is not None else "—",
                "Severity": e.severity.upper(), "Conf": round(e.confidence, 2),
                "Risk": e.risk_score, "Details": e.details,
            })

        risk_index = max((e.risk_score for e in self.recent_events
                          if now - e.ts < 30), default=0.0)
        if now - self._last_hist > 0.5:
            self.risk_history.append((round(now - self.start_time, 1), risk_index))
            self._last_hist = now

        self.frames += 1
        ann = self._annotate(frame, tracks, scenario, risk_index, now)
        return ann, self._stats(tracks, doors, scenario, risk_index, now)

    def _annotate(self, img, tracks, scenario, risk_index, now):
        H, W = img.shape[:2]
        out = img
        if self.show_zones:
            overlay = out.copy()
            for z in CONFIG["zones"]:
                zp = self.zones.polys.get(z["id"])
                if zp is None:
                    continue
                cv2.fillPoly(overlay, [zp], z["color"])
                cv2.polylines(out, [zp], True, z["color"], 2)
            out = cv2.addWeighted(overlay, 0.16, out, 0.84, 0)
            for z in CONFIG["zones"]:
                zp = self.zones.polys[z["id"]]
                x, y = int(zp[:, 0].min()), int(zp[:, 1].min())
                cv2.putText(out, z["label"], (x + 6, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, z["color"], 2, cv2.LINE_AA)

        if self.show_tracks:
            for tr in tracks:
                x1, y1, x2, y2 = [int(v) for v in tr.bbox]
                col = CLASS_COLORS.get(tr.cls, (160, 160, 160))
                cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
                zone_txt = f" | {tr.zone}" if tr.zone else ""
                label = f"#{tr.id} {tr.cls} {tr.conf:.2f}{zone_txt}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
                cv2.putText(out, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

        hot = [e for e in self.recent_events
               if now - e.ts < 2.5 and e.severity in ("high", "critical")]
        if hot:
            pulse = int(3 + 3 * abs(np.sin(now * 5)))
            col = (60, 60, 255) if max(e.risk_score for e in hot) >= 80 else (40, 140, 255)
            cv2.rectangle(out, (2, 2), (W - 3, H - 3), col, pulse)

        if self.show_hud:
            cv2.rectangle(out, (0, 0), (W, 36), (28, 22, 16), -1)
            stamp = datetime.now().strftime("%H:%M:%S.") + f"{int((now % 1) * 100):02d}"
            rec = int(now * 2) % 2 == 0
            cv2.circle(out, (18, 18), 6, (60, 60, 255) if rec else (90, 90, 160), -1)
            hud = (f"{self.camera}  |  {scenario}  |  {stamp}  |  "
                   f"FRAME {self.frames:06d}  |  FPS {self.fps_ema:4.1f}  |  "
                   f"TRACKS {len(tracks)}  |  RISK {risk_index:4.1f}")
            cv2.putText(out, hud, (34, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (235, 235, 235), 1, cv2.LINE_AA)
        return out

    def _stats(self, tracks, doors, scenario, risk_index, now):
        counts = {}
        for t in tracks:
            counts[t.cls] = counts.get(t.cls, 0) + 1
        active = [e for e in self.recent_events
                  if now - e.ts < CONFIG["active_event_window_s"]]
        doors_open = sum(1 for d in doors.values() if d["open"]) if doors else 0
        return {
            "fps": self.fps_ema, "frames": self.frames, "tracks": len(tracks),
            "counts": counts, "active_events": active, "risk_index": risk_index,
            "scenario": scenario, "doors_open": doors_open,
            "uptime": time.time() - self.start_time,
            "detector_status": self.vision.detector_status,
            "tracker_status": f"{len(tracks)} active / {self.tracker.total_ids} total IDs",
            "db_status": f"{self.reporter.mode} — {self.reporter.rows} rows",
            "event_count": self.events_agent.emitted,
            "inv_status": self.investigator.status,
            "inv_count": self.investigator.processed,
            "latest_analysis": (self.investigator.analyses[0]
                                if self.investigator.analyses else None),
        }


def record_demo_clip(seconds=8):
    clip_pipe = Pipeline("demo", camera="CAM-01 // DEMO CLIP")
    path = os.path.join(tempfile.gettempdir(), f"dcvision_demo_{uuid.uuid4().hex[:6]}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             CONFIG["demo_fps"], CONFIG["demo_size"])
    n = int(seconds * CONFIG["demo_fps"])
    for _ in range(n):
        frame, _ = clip_pipe.step()
        writer.write(frame)
        time.sleep(1.0 / CONFIG["demo_fps"])
    writer.release()
    clip_pipe.vision.release()
    return path


# --------------------------------------------------------------------------
# UI HELPERS
# --------------------------------------------------------------------------
def inject_css():
    st.markdown("""
    <style>
      .stApp { background:#0b0f17; }
      div[data-testid="stMetric"] {
        background:#121826; border:1px solid #1f2a3d; border-radius:12px;
        padding:10px 14px;
      }
      div[data-testid="stMetricLabel"] { color:#8ea0b8; }
      .ev-card {
        border-radius:10px; padding:10px 12px; margin:6px 0;
        border:1px solid #243044; background:#111827;
        font-size:0.86rem; color:#dbe4f0;
      }
      .ev-badge {
        display:inline-block; padding:1px 9px; border-radius:999px;
        font-size:0.7rem; font-weight:700; letter-spacing:.04em;
        margin-right:6px; vertical-align:middle;
      }
      .ai-card {
        background:linear-gradient(135deg,#0f172a,#111c33);
        border:1px solid #274060; border-radius:12px;
        padding:12px 14px; color:#d7e3f4; font-size:0.85rem;
      }
      .pill { padding:2px 10px; border-radius:999px; font-size:.72rem; font-weight:700; }
      .pill-live { background:#052e16; color:#4ade80; border:1px solid #14532d; }
      .pill-demo { background:#1e1b4b; color:#a5b4fc; border:1px solid #3730a3; }
      .small-dim { color:#7d8ca3; font-size:.75rem; }
    </style>""", unsafe_allow_html=True)


def event_card_html(e: VisionEvent, age: float) -> str:
    bg, fg = SEV_COLOR.get(e.severity, SEV_COLOR["medium"])
    return f"""
    <div class="ev-card">
      <span class="ev-badge" style="background:{bg};color:{fg};">{e.severity.upper()}</span>
      <b>{EVENT_LABELS.get(e.event_type, e.event_type)}</b>
      <span class="small-dim"> · {e.zone or '—'} · {int(age)}s ago · risk {e.risk_score}</span>
      <div class="small-dim" style="margin-top:3px;">{e.details}</div>
    </div>"""


SCHEMA_JSON = {
    "event_id": "evt_9f2c1a", "ts": "2026-08-15T14:03:22Z", "camera": "CAM-01",
    "event_type": "unauthorized_zone_entry", "track_id": 14,
    "object_class": "technician", "zone": "restricted_area", "severity": "high",
    "confidence": 0.87, "risk_score": 78.4,
    "details": "Person entered restricted_area; credential verification required.",
    "recommended_response": "Dispatch floor operator to verify badge; audit door logs.",
}

DDL = """
CREATE TABLE vision_events (
    id           SERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL,
    camera       VARCHAR(64),
    track_id     INTEGER,
    object_class VARCHAR(64),
    zone         VARCHAR(64),
    event_type   VARCHAR(64),
    severity     VARCHAR(16),
    confidence   FLOAT,
    risk_score   FLOAT,
    details      TEXT
);
CREATE INDEX ix_vision_events_ts ON vision_events(ts);
CREATE INDEX ix_vision_events_type ON vision_events(event_type);
"""


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="DC Vision Intelligence", page_icon="🛰️",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()

    with st.sidebar:
        st.markdown("### 🛰️ Mission Control")
        source = st.radio("Video source",
                          ["Demo simulation (live)", "Webcam", "Upload video",
                           "RTSP / IP camera"], index=0)
        url_in, up_file = "", None
        if source == "RTSP / IP camera":
            url_in = st.text_input("RTSP URL",
                                   placeholder="rtsp://user:pass@host:554/stream")
        if source == "Upload video":
            up_file = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])

        st.divider()
        sz = st.session_state
        sz.setdefault("show_zones", True)
        sz.setdefault("show_tracks", True)
        sz.setdefault("show_hud", True)
        sz["show_zones"] = st.toggle("Virtual zones overlay", value=sz["show_zones"])
        sz["show_tracks"] = st.toggle("Detection boxes + track IDs",
                                      value=sz["show_tracks"])
        sz["show_hud"] = st.toggle("Camera HUD", value=sz["show_hud"])
        sz["demo_speed"] = st.slider("Demo motion speed", 0.5, 3.0, 1.5, 0.25)

        st.divider()
        running = sz.get("running", True)
        if st.button("⏹  Stop pipeline" if running else "▶  Start pipeline",
                     use_container_width=True):
            sz["running"] = not running
            st.rerun()

        if st.button("🎬  Export demo clip (MP4)", use_container_width=True):
            with st.spinner("Recording live demo feed for 8 seconds…"):
                clip_path = record_demo_clip(8)
            with open(clip_path, "rb") as fh:
                clip_bytes = fh.read()
            st.download_button("⬇  Download datacenter_demo.mp4", clip_bytes,
                               file_name="datacenter_demo.mp4", mime="video/mp4",
                               use_container_width=True)

        st.caption("Detections are observations — never identity or access-control "
                   "decisions. Human oversight required.")

    # ---------------- pipeline (re)build ----------------
    sz = st.session_state
    sz.setdefault("running", True)

    if source.startswith("Demo"):
        desired = ("demo", "", "")
    elif source == "Webcam":
        desired = ("webcam", "", "")
    elif source == "Upload video" and up_file is not None:
        fid = getattr(up_file, "file_id", up_file.name)
        key = f"upload_path_{fid}"
        if key not in sz:
            tmp = os.path.join(tempfile.gettempdir(),
                               f"dcvision_{uuid.uuid4().hex[:6]}_{up_file.name}")
            with open(tmp, "wb") as fh:
                fh.write(up_file.getbuffer())
            sz[key] = tmp
        desired = ("file", sz[key], "")
    elif source == "Upload video":
        desired = ("demo", "", "")
    else:
        desired = ("rtsp", "", url_in)

    fallback_note = None
    if sz.get("source_key") != desired or "pipeline" not in sz:
        try:
            stype, path, url = desired
            sz["pipeline"] = Pipeline(stype, path=path, url=url)
            sz["source_key"] = desired
            sz["running"] = True
        except Exception as e:
            fallback_note = str(e)
            sz["pipeline"] = Pipeline("demo")
            sz["source_key"] = ("demo", "", "")
            sz["running"] = True

    pipe: Pipeline = sz["pipeline"]
    pipe.show_zones = sz["show_zones"]
    pipe.show_tracks = sz["show_tracks"]
    pipe.show_hud = sz["show_hud"]
    if pipe.vision.scene is not None:
        pipe.vision.scene.speed_mult = float(sz.get("demo_speed", 1.5))

    mode = "DEMO SIMULATION" if pipe.vision.scene is not None else "LIVE SOURCE"
    pill = "pill-demo" if pipe.vision.scene is not None else "pill-live"
    h1, h2 = st.columns([5, 1.2])
    h1.markdown(f"## 🏢 {CONFIG['app_title']}")
    h1.caption(f"{CONFIG['org']} · multi-agent vision operations · "
               f"{datetime.now().strftime('%A, %B %d %Y')}")
    h2.markdown(
        f"<div style='text-align:right;margin-top:10px;'>"
        f"<span class='pill {pill}'>● {mode}</span></div>", unsafe_allow_html=True)
    if fallback_note:
        st.warning(f"⚠️ Requested source unavailable ({fallback_note}) — "
                   f"auto-switched to live demo simulation.")

    m = st.columns(6)
    ph = {"fps": m[0].empty(), "tracks": m[1].empty(), "events": m[2].empty(),
          "risk": m[3].empty(), "doors": m[4].empty(), "uptime": m[5].empty()}

    col_vid, col_side = st.columns([2.15, 1], gap="large")
    vid_ph = col_vid.empty()
    live_ph = col_vid.empty()
    with col_side:
        st.markdown("#### 🚨 Active Events")
        ev_ph = st.empty()
        st.markdown("#### 🧠 Investigation Agent")
        ai_ph = st.empty()
        st.markdown("#### 📦 Object Counters")
        cnt_ph = st.empty()

    tab_tl, tab_an, tab_sys, tab_schema = st.tabs(
        ["🗂 Event Timeline", "📈 Analytics", "🖥 System & Agents", "🧾 Event Schema"])
    with tab_tl:
        tl_ph = st.empty()
    with tab_an:
        ch_ph = st.empty()
    with tab_sys:
        sys_ph = st.empty()
    with tab_schema:
        st.markdown("**Structured event message (agent bus / DB row):**")
        st.json(SCHEMA_JSON)
        st.markdown("**PostgreSQL DDL (auto-created on boot via SQLAlchemy):**")
        st.code(DDL, language="sql")

    # ---------------- live loop (exception-safe) ----------------
    loop, consec_err = 0, 0
    while sz.get("running", True):
        try:
            frame, stats = pipe.step()
            loop += 1
            consec_err = 0

            vid_ph.image(frame, channels="BGR", use_container_width=True)
            live_ph.caption(f"🟢 Streaming — frame **{stats['frames']}** · "
                            f"scenario: **{stats['scenario']}** · "
                            f"last update {datetime.now().strftime('%H:%M:%S')}")

            ph["fps"].metric("FPS", f"{stats['fps']:.1f}")
            ph["tracks"].metric("Active Tracks", stats["tracks"])
            ph["events"].metric("Active Events", len(stats["active_events"]))
            ph["risk"].metric("Risk Index", f"{stats['risk_index']:.0f} / 100")
            ph["doors"].metric("Doors Open", stats["doors_open"])
            mm, ss = divmod(int(stats["uptime"]), 60)
            ph["uptime"].metric("Uptime", f"{mm:02d}:{ss:02d}")

            evs = sorted(stats["active_events"], key=lambda e: -e.risk_score)[:6]
            if evs:
                now = time.time()
                ev_ph.markdown("".join(event_card_html(e, now - e.ts) for e in evs),
                               unsafe_allow_html=True)
            else:
                ev_ph.markdown("<div class='ev-card'>✅ No active events — floor nominal.</div>",
                               unsafe_allow_html=True)

            a = stats["latest_analysis"]
            if a:
                bg, fg = SEV_COLOR.get(a["severity"], SEV_COLOR["medium"])
                ai_ph.markdown(f"""
                <div class="ai-card">
                  <span class="ev-badge" style="background:{bg};color:{fg};">{a['severity'].upper()}</span>
                  <b>{a['event']}</b> · <span class="small-dim">{a['zone']} · risk {a['risk']}</span><br><br>
                  <b>Assessment:</b> {a['assessment']}<br><br>
                  <b>Recommended response:</b> {a['response']}<br><br>
                  <span class="small-dim">status: {a['status']} · agent: {stats['inv_status']} ·
                  analyses: {stats['inv_count']}</span>
                </div>""", unsafe_allow_html=True)
            else:
                ai_ph.markdown("<div class='ai-card'>🧠 Investigation agent idle — monitoring "
                               "structured event bus…</div>", unsafe_allow_html=True)

            chips = " &nbsp; ".join(
                f"<span class='pill' style='background:#12203a;color:#9ec5ff;"
                f"border:1px solid #24406b;'>{k}: {v}</span>"
                for k, v in stats["counts"].items()) or \
                "<span class='small-dim'>no tracked objects</span>"
            cnt_ph.markdown(chips, unsafe_allow_html=True)

            if loop % 12 == 1:
                if pipe.timeline:
                    tl_ph.dataframe(pd.DataFrame(list(pipe.timeline)[:200]),
                                    use_container_width=True, height=280)
                if len(pipe.risk_history) > 2:
                    df = pd.DataFrame(list(pipe.risk_history), columns=["t", "risk"])
                    ch_ph.line_chart(df.set_index("t"))
                sys_ph.markdown(f"""
                | Component | Status |
                |---|---|
                | Vision Agent | {stats['detector_status']} · source: {mode} |
                | Tracking Agent | ByteTrack-style IoU + motion model · {stats['tracker_status']} |
                | Zone Agent | {len(CONFIG['zones'])} virtual zones active |
                | Event Agent | {stats['event_count']} events emitted |
                | Risk Agent | current index {stats['risk_index']:.0f}/100 |
                | Investigation Agent | {stats['inv_status']} · {stats['inv_count']} analyses |
                | Reporting Agent | {stats['db_status']} |
                | Frames processed | {stats['frames']} |
                """)

            time.sleep(1.0 / CONFIG["demo_fps"] if pipe.vision.scene is not None else 0.01)

        except Exception as e:
            consec_err += 1
            vid_ph.error(f"Pipeline error #{consec_err}: {e}")
            st.code(traceback.format_exc())
            if consec_err >= 10:
                st.error("Pipeline stopped after repeated errors. Reload the page to restart.")
                break
            time.sleep(0.5)

    if not sz.get("running", True):
        st.info("Pipeline paused — press **Start pipeline** in the sidebar to resume.")


main()