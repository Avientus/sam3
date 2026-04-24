#!/usr/bin/env python3
"""
SAM3 ROS2 Node

Subscribes to a camera image and a text prompt, forwards frames to the
SAM3 inference server (REST API), and publishes the results.

Two trigger modes (parameter: trigger_mode):
  "prompt"     — run inference only when a new /sam3/prompt message arrives
  "continuous" — run inference continuously at throttle_hz (default)

Subscribed topics:
  /sam3/image    sensor_msgs/Image   — camera frames
  /sam3/prompt   std_msgs/String     — text prompt  (triggers inference in "prompt" mode)

Published topics:
  /sam3/detections        vision_msgs/Detection2DArray  — bounding boxes + scores
  /sam3/image_annotated   sensor_msgs/Image             — frame with masks overlaid
  /sam3/result_json       std_msgs/String               — raw server JSON response

Parameters:
  server_url    (string,  default "http://localhost:8000")
  text_prompt   (string,  default "person")
  include_masks (bool,    default True)
  trigger_mode  (string,  default "continuous")  — "continuous" or "prompt"
  throttle_hz   (double,  default 2.0)           — max Hz in continuous mode
"""

import base64
import json
import threading
import time
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255),
    (255, 0, 128), (128, 0, 255), (0, 255, 128),
]


class SAM3Node(Node):
    def __init__(self):
        super().__init__("sam3_node")

        self.declare_parameter("server_url",   "http://localhost:8000")
        self.declare_parameter("text_prompt",  "person")
        self.declare_parameter("include_masks", True)
        self.declare_parameter("trigger_mode", "continuous")  # "continuous" | "prompt"
        self.declare_parameter("throttle_hz",  2.0)

        self._server_url   = self.get_parameter("server_url").value
        self._prompt       = self.get_parameter("text_prompt").value
        self._include_masks = self.get_parameter("include_masks").value
        self._trigger_mode = self.get_parameter("trigger_mode").value
        throttle_hz        = self.get_parameter("throttle_hz").value

        self._min_interval = 1.0 / max(throttle_hz, 0.01)
        self._last_call    = 0.0
        self._busy         = False
        self._lock         = threading.Lock()

        self._latest_img_msg = None   # most recent buffered image
        self._img_lock       = threading.Lock()

        self._bridge = CvBridge()

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(Image,  "/sam3/image",  self._image_cb,  10)
        self.create_subscription(String, "/sam3/prompt", self._prompt_cb, 10)

        # ── Publishers ──────────────────────────────────────────────────────
        self._pub_detections = self.create_publisher(Detection2DArray, "/sam3/detections",      10)
        self._pub_annotated  = self.create_publisher(Image,            "/sam3/image_annotated", 10)
        self._pub_json       = self.create_publisher(String,           "/sam3/result_json",     10)

        self.get_logger().info(
            f"SAM3 node ready | server: {self._server_url} | "
            f"mode: {self._trigger_mode} | prompt: '{self._prompt}'"
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        with self._img_lock:
            self._latest_img_msg = msg

        if self._trigger_mode == "continuous":
            now = time.monotonic()
            with self._lock:
                if self._busy or (now - self._last_call) < self._min_interval:
                    return
                self._busy     = True
                self._last_call = now
            threading.Thread(target=self._infer, args=(msg,), daemon=True).start()

    def _prompt_cb(self, msg: String):
        self._prompt = msg.data
        self.get_logger().info(f"Prompt updated: '{self._prompt}'")

        if self._trigger_mode == "prompt":
            with self._img_lock:
                img_msg = self._latest_img_msg

            if img_msg is None:
                self.get_logger().warn("Prompt received but no image buffered yet on /sam3/image")
                return

            with self._lock:
                if self._busy:
                    self.get_logger().warn("Prompt received but inference already running, skipping")
                    return
                self._busy = True

            threading.Thread(target=self._infer, args=(img_msg,), daemon=True).start()

    # ── Inference thread ─────────────────────────────────────────────────────

    def _infer(self, msg: Image):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            _, jpg  = cv2.imencode(".jpg", cv_img)

            response = requests.post(
                f"{self._server_url}/predict/image",
                files={"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
                data={
                    "text_prompt":   self._prompt,
                    "include_masks": str(self._include_masks).lower(),
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

            n = len(result.get("detections", []))
            self.get_logger().info(
                f"'{self._prompt}' → {n} detection(s)  [{result.get('inference_time_s', '?')}s]"
            )
        except Exception as exc:
            self.get_logger().error(f"SAM3 request failed: {exc}")
            return
        finally:
            with self._lock:
                self._busy = False

        self._publish_json(result)
        self._publish_detections(msg, result)
        self._publish_annotated(msg, cv_img, result)

    # ── Publishers ───────────────────────────────────────────────────────────

    def _publish_json(self, result: dict):
        msg = String()
        msg.data = json.dumps(result)
        self._pub_json.publish(msg)

    def _publish_detections(self, header_msg: Image, result: dict):
        array        = Detection2DArray()
        array.header = header_msg.header

        for det in result.get("detections", []):
            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])

            d        = Detection2D()
            d.header = header_msg.header
            d.bbox.center.position.x = (bbox[0] + bbox[2]) / 2.0
            d.bbox.center.position.y = (bbox[1] + bbox[3]) / 2.0
            d.bbox.size_x = float(bbox[2] - bbox[0])
            d.bbox.size_y = float(bbox[3] - bbox[1])

            hyp                      = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id  = str(det.get("class_id", -1))
            hyp.hypothesis.score     = float(det.get("score", 0.0))
            d.results.append(hyp)

            array.detections.append(d)

        self._pub_detections.publish(array)

    def _publish_annotated(self, header_msg: Image, cv_img: np.ndarray, result: dict):
        annotated = cv_img.copy()

        for i, det in enumerate(result.get("detections", [])):
            color = COLORS[i % len(COLORS)]

            if "mask_b64" in det:
                try:
                    mask = np.array(
                        PILImage.open(BytesIO(base64.b64decode(det["mask_b64"]))).convert("L")
                    )
                    overlay = np.zeros_like(annotated)
                    overlay[mask > 128] = color
                    annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.4, 0)
                except Exception as exc:
                    self.get_logger().debug(f"Mask decode error: {exc}")

            bbox  = det.get("bbox_xyxy", [0, 0, 0, 0])
            score = det.get("score", 0.0)
            cv2.rectangle(annotated,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, 2)
            cv2.putText(annotated,
                        f"{self._prompt} {score:.2f}",
                        (int(bbox[0]), max(int(bbox[1]) - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        out_msg        = self._bridge.cv2_to_imgmsg(annotated, "bgr8")
        out_msg.header = header_msg.header
        self._pub_annotated.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SAM3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
