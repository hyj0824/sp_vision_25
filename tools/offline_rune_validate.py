#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml


ANGLE_BETWEEN_FAN_BLADES = math.radians(72.0)


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def rotation_matrix(yaw, pitch, roll=0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def read_matrix(cfg, key, rows, cols):
    return np.array(cfg[key], dtype=np.float64).reshape(rows, cols)


def read_vector(cfg, key):
    return np.array(cfg[key], dtype=np.float64).reshape(-1)


def limit_rad(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


class OfflineRunePredictor:
    def __init__(self, cfg):
        self.camera_matrix = read_matrix(cfg, "camera_matrix", 3, 3)
        self.distort = read_vector(cfg, "distort_coeffs")
        self.R_gimbal2imubody = read_matrix(cfg, "R_gimbal2imubody", 3, 3)
        self.R_camera2gimbal = read_matrix(cfg, "R_camera2gimbal", 3, 3)
        self.t_camera2gimbal = read_vector(cfg, "t_camera2gimbal")
        self.R_gimbal2world = np.eye(3, dtype=np.float64)

        self.yaw_offset = math.radians(float(cfg.get("rune_yaw_offset", cfg.get("yaw_offset", 0.0))))
        self.pitch_offset = math.radians(float(cfg.get("rune_pitch_offset", cfg.get("pitch_offset", 0.0))))
        self.delay = float(cfg.get("rune_delay_time", cfg.get("predict_time", 0.1)))
        self.target_radius = float(cfg.get("rune_target_radius", cfg.get("target_radius", 0.145)))
        self.fan_len = float(cfg.get("rune_fan_len", cfg.get("fan_len", 0.7)))
        self.small_speed = float(cfg.get("rune_small_speed", math.pi / 3.0))
        self.direction_window = int(cfg.get("rune_direction_window", 10))

        self.first = True
        self.R_base = np.eye(3, dtype=np.float64)
        self.angle_abs_last = 0.0
        self.total_shift = 0
        self.angle_rel = 0.0
        self.direction_data = []
        self.direction = 0

    def set_gimbal_yaw_pitch(self, yaw, pitch):
        R_imu = rotation_matrix(yaw, pitch, 0.0)
        self.R_gimbal2world = self.R_gimbal2imubody.T @ R_imu @ self.R_gimbal2imubody

    def solve_pnp(self, keypoints):
        object_points = np.array(
            [
                [0.0, +self.target_radius, 0.0],
                [-self.target_radius, 0.0, 0.0],
                [0.0, -self.target_radius, 0.0],
                [+self.target_radius, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            np.asarray(keypoints, dtype=np.float64),
            self.camera_matrix,
            self.distort,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return None

        R_target2camera, _ = cv2.Rodrigues(rvec)
        target_camera = tvec.reshape(3)
        target_gimbal = self.R_camera2gimbal @ target_camera + self.t_camera2gimbal
        target_world = self.R_gimbal2world @ target_gimbal
        R_target2world = self.R_gimbal2world @ self.R_camera2gimbal @ R_target2camera

        if self.first:
            self.first = False
            self.R_base = R_target2world.copy()
            self.angle_abs_last = 0.0
            self.total_shift = 0

        R_rel = self.R_base.T @ R_target2world
        angle_abs = -math.atan2(R_rel[1, 0], R_rel[0, 0])
        delta_abs = angle_abs - self.angle_abs_last
        self.angle_abs_last = angle_abs
        self.total_shift += int(round(delta_abs / ANGLE_BETWEEN_FAN_BLADES))
        self.angle_rel = angle_abs - self.total_shift * ANGLE_BETWEEN_FAN_BLADES

        self._update_direction(self.angle_rel)

        return {
            "target_camera": target_camera,
            "target_world": target_world,
            "R_target2world": R_target2world,
            "distance": float(np.linalg.norm(target_world)),
            "rvec": rvec,
            "tvec": tvec,
            "object_points": object_points,
        }

    def _update_direction(self, angle):
        if self.direction != 0:
            return
        self.direction_data.append(angle)
        if len(self.direction_data) < max(2, self.direction_window):
            return
        half = len(self.direction_data) // 2
        stable = anti = clockwise = 0
        for i in range(half):
            diff = self.direction_data[i + half] - self.direction_data[i]
            if diff > 1.5e-2:
                clockwise += 1
            elif diff < -1.5e-2:
                anti += 1
            else:
                stable += 1
        best = max(stable, anti, clockwise)
        if best == clockwise:
            self.direction = 1
        elif best == anti:
            self.direction = -1
        else:
            self.direction = 2

    def aim(self, observation, bullet_speed, fps):
        if observation is None:
            return None
        if bullet_speed < 10.0 or bullet_speed > 35.0:
            bullet_speed = 24.0

        direction_sign = 0.0
        if self.direction == 1:
            direction_sign = 1.0
        elif self.direction == -1:
            direction_sign = -1.0

        target_world = observation["target_world"]
        fly_time = self._fly_time(target_world, bullet_speed)
        future_dt = (fly_time if fly_time is not None else 0.0) + self.delay + 1.0 / max(fps, 1.0)
        rotation = direction_sign * self.small_speed * future_dt
        aim_world = self.predict_point(observation, rotation)
        fly_time = self._fly_time(aim_world, bullet_speed)
        if fly_time is None:
            return None
        yaw = math.atan2(aim_world[1], aim_world[0]) + self.yaw_offset
        pitch = -(self._trajectory_pitch(aim_world, bullet_speed) + self.pitch_offset)
        return {
            "yaw": yaw,
            "pitch": pitch,
            "rotation": rotation,
            "fly_time": fly_time,
            "aim_world": aim_world,
        }

    def predict_point(self, observation, rotation):
        shift = np.array(
            [
                self.fan_len * math.sin(rotation),
                self.fan_len - self.fan_len * math.cos(rotation),
                0.0,
            ],
            dtype=np.float64,
        )
        return observation["R_target2world"] @ shift + observation["target_world"]

    def project_world_point(self, observation, point_world):
        R_target2camera = self.R_camera2gimbal.T @ self.R_gimbal2world.T @ observation["R_target2world"]
        target_camera = self.R_camera2gimbal.T @ (
            self.R_gimbal2world.T @ point_world - self.t_camera2gimbal
        )
        rvec, _ = cv2.Rodrigues(R_target2camera)
        image_points, _ = cv2.projectPoints(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            rvec,
            target_camera.reshape(3, 1),
            self.camera_matrix,
            self.distort,
        )
        return tuple(image_points.reshape(-1, 2)[0])

    def _trajectory_pitch(self, xyz, bullet_speed):
        d = math.hypot(float(xyz[0]), float(xyz[1]))
        h = float(xyz[2])
        g = 9.7833
        a = g * d * d / (2.0 * bullet_speed * bullet_speed)
        b = -d
        c = a + h
        delta = b * b - 4.0 * a * c
        if delta < 0.0:
            return float("nan")
        tan1 = (-b + math.sqrt(delta)) / (2.0 * a)
        tan2 = (-b - math.sqrt(delta)) / (2.0 * a)
        p1 = math.atan(tan1)
        p2 = math.atan(tan2)
        t1 = d / (bullet_speed * math.cos(p1))
        t2 = d / (bullet_speed * math.cos(p2))
        return p1 if t1 < t2 else p2

    def _fly_time(self, xyz, bullet_speed):
        pitch = self._trajectory_pitch(xyz, bullet_speed)
        if not math.isfinite(pitch):
            return None
        d = math.hypot(float(xyz[0]), float(xyz[1]))
        return d / (bullet_speed * math.cos(pitch))


class OnnxRuneDetector:
    def __init__(self, model_path, conf_threshold):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        x = cv2.cvtColor(cv2.resize(bgr, (640, 384)), cv2.COLOR_BGR2RGB).astype(np.float32)[None]
        raw = self.session.run(None, {self.input_name: x})[0].reshape(-1, 22)
        raw_conf = raw[:, 8]
        best = int(np.argmax(raw_conf))
        conf = sigmoid(float(raw_conf[best]))
        pts = raw[best, :8].reshape(4, 2).astype(np.float64)
        pts[:, 0] *= w / 640.0
        pts[:, 1] *= h / 384.0
        if conf < self.conf_threshold:
            return None, conf, pts
        return pts, conf, pts


def draw_overlay(img, frame_idx, conf, pts, observation, aim, predictor):
    if pts is not None:
        pts_int = np.round(pts).astype(int)
        for i, p in enumerate(pts_int):
            cv2.circle(img, tuple(p), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i), tuple(p + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.polylines(img, [pts_int.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    if observation is not None:
        center = np.mean(pts, axis=0).astype(int)
        cv2.circle(img, tuple(center), 4, (255, 0, 0), -1)
    if observation is not None and aim is not None:
        px = predictor.project_world_point(observation, aim["aim_world"])
        p = (int(round(px[0])), int(round(px[1])))
        cv2.drawMarker(img, p, (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
    text = f"frame={frame_idx} conf={conf:.3f}"
    if observation is not None:
        text += f" dist={observation['distance']:.2f} angle={math.degrees(predictor.angle_rel):.1f}"
    if aim is not None:
        text += f" yaw={math.degrees(aim['yaw']):.2f} pitch={math.degrees(aim['pitch']):.2f}"
    cv2.putText(img, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/standard3.yaml")
    parser.add_argument("--model", default="assets/2025-07-28-rune.onnx")
    parser.add_argument("--out-dir", default="outputs/rune_offline")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-every", type=int, default=120)
    parser.add_argument("--bullet-speed", type=float, default=24.0)
    parser.add_argument("--pose-mode", choices=["feedback", "identity"], default="feedback")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    detector = OnnxRuneDetector(args.model, float(cfg.get("rune_confidence_threshold", 0.7)))
    predictor = OfflineRunePredictor(cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps / max(args.stride, 1), (width, height))

    csv_path = out_dir / "rune_offline.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame", "time", "conf", "valid", "distance", "cam_x", "cam_y", "cam_z",
            "world_x", "world_y", "world_z", "angle_deg", "direction", "yaw_deg",
            "pitch_deg", "predict_rotation_deg", "fly_time",
            "p0x", "p0y", "p1x", "p1y", "p2x", "p2y", "p3x", "p3y",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()

        yaw_feedback = 0.0
        pitch_feedback = 0.0
        processed = valid = controlled = 0
        frame_idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            if args.pose_mode == "feedback":
                predictor.set_gimbal_yaw_pitch(yaw_feedback, pitch_feedback)
            else:
                predictor.set_gimbal_yaw_pitch(0.0, 0.0)

            pts, conf, raw_pts = detector.detect(img)
            observation = predictor.solve_pnp(pts) if pts is not None else None
            aim = predictor.aim(observation, args.bullet_speed, fps)
            if aim is not None:
                controlled += 1
                yaw_feedback = aim["yaw"]
                pitch_feedback = aim["pitch"]
            if observation is not None:
                valid += 1

            row = {
                "frame": frame_idx,
                "time": frame_idx / fps,
                "conf": conf,
                "valid": int(observation is not None),
                "distance": observation["distance"] if observation is not None else "",
                "cam_x": observation["target_camera"][0] if observation is not None else "",
                "cam_y": observation["target_camera"][1] if observation is not None else "",
                "cam_z": observation["target_camera"][2] if observation is not None else "",
                "world_x": observation["target_world"][0] if observation is not None else "",
                "world_y": observation["target_world"][1] if observation is not None else "",
                "world_z": observation["target_world"][2] if observation is not None else "",
                "angle_deg": math.degrees(predictor.angle_rel) if observation is not None else "",
                "direction": predictor.direction,
                "yaw_deg": math.degrees(aim["yaw"]) if aim is not None else "",
                "pitch_deg": math.degrees(aim["pitch"]) if aim is not None else "",
                "predict_rotation_deg": math.degrees(aim["rotation"]) if aim is not None else "",
                "fly_time": aim["fly_time"] if aim is not None else "",
            }
            draw_pts = pts if pts is not None else raw_pts
            if draw_pts is not None:
                for i in range(4):
                    row[f"p{i}x"] = draw_pts[i, 0]
                    row[f"p{i}y"] = draw_pts[i, 1]
            writer_csv.writerow(row)

            if processed % args.save_every == 0:
                vis = img.copy()
                draw_overlay(vis, frame_idx, conf, pts, observation, aim, predictor)
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), vis)
            if writer is not None:
                vis = img.copy()
                draw_overlay(vis, frame_idx, conf, pts, observation, aim, predictor)
                writer.write(vis)

            processed += 1
            if processed % 100 == 0:
                print(f"processed={processed} valid={valid} controlled={controlled}")
            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    print(f"done processed={processed} valid={valid} controlled={controlled}")
    print(f"csv={csv_path}")
    print(f"frames={frames_dir}")
    if args.save_video:
        print(f"video={out_dir / 'overlay.mp4'}")


if __name__ == "__main__":
    main()
