#!/usr/bin/env python3
"""
webcam_test.py - Live webcam segmentation using SAM 2.1 (tiny) from Hugging Face.

Usage:
    python visual/webcam_test.py

Requirements:
    pip install torch torchvision transformers opencv-python pillow

Controls:
    - Left-click on the webcam feed to add a foreground point.
    - Right-click to add a background point.
    - Press 'r' to reset points and masks.
    - Press 'q' or ESC to quit.
"""

import sys
import os
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor


def load_model(device: str = "cuda"):
    """Load the SAM 2.1 tiny model and processor from Hugging Face."""
    model_id = "facebook/sam2.1-hiera-tiny"
    print(f"Loading model '{model_id}' ...")
    processor = Sam2VideoProcessor.from_pretrained(model_id)
    model = Sam2VideoModel.from_pretrained(model_id).to(device)
    model.eval()
    print("Model loaded successfully.")
    return model, processor

def segment_from_points(
    model: Sam2VideoModel, 
    processor: Sam2VideoProcessor, 
    session, frame: np.ndarray, 
    device: str = "cuda"
):
    # Run SAM inference given a set of click points and their labels.
    # Returns the mask as a numpy array (H, W) of bools.

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # input_points = [[points]]
    # input_labels = [[labels]]

    inputs = processor(
        pil_image,    
        # input_points=input_points,
        # input_labels=input_labels,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(inference_session=session, frame=inputs.pixel_values[0])

    # Post-process to get the mask in the original image size
    video_masks = processor.post_process_masks(
        [outputs.pred_masks.cpu()],
        inputs["original_sizes"].cpu(),
        #inputs["reshaped_input_size"].cpu(),
    )

    #print(video_masks.shape)
    # Take the highest-confidence mask (index 0, batch 0)
    mask = video_masks[0][0].numpy()  # shape: (num_masks, H, W)
    #best_mask = mask[0]  # highest confidence
    print(mask.shape)
    return mask[0]


def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5):
    
    
    overlay = frame.copy()
    overlay[mask] = color

    cv2.imshow('mask', overlay)
    cv2.waitKey(500)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_points(frame: np.ndarray, points: list, labels: list):
    for (x, y), lbl in zip(points, labels):
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        cv2.circle(frame, (int(x), int(y)), 7, (255, 255, 255), 1)


def main():
    # --- Configuration -----------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CAMERA_INDEX = 0
    CAM_WIDTH = 640
    CAM_HEIGHT = 480

    print(f"Using device: {DEVICE}")

    # Load model
    model, processor = load_model(device=DEVICE)

    # Start inference session
    inference_session = processor.init_video_session(
        inference_device=DEVICE,
    )

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    print("Webcam opened. Controls:")
    print("  Left-click  = add foreground point")
    print("  Right-click = add background point")
    print("  'r'         = reset points/mask")
    print("  'q' / ESC   = quit")

    # State
    points = []   # list of [x, y]
    labels = []   # list of 1 (fg) or 0 (bg)
    needs_update = False
    ready = False

    # Mouse callback
    def on_mouse(event, x, y, flags, param):
        nonlocal needs_update, ready
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            labels.append(1)  # foreground
            needs_update = True
            ready = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y])
            labels.append(0)  # background
            needs_update = True
            ready = True

    window_name = "SAM 2.1 Tiny - Webcam Segmentation"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    # Process frames one by one
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        frame_idx += 1
        if not ret:
            print("WARNING: Failed to read frame from camera.")
            break

        display = frame.copy()

        if needs_update:
            processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=frame_idx,
            obj_ids=1,
            input_points= [[points]], 
            input_labels= [[labels]], 
            original_size=frame.shape
            )
            needs_update = False

        if ready:
            print(f"Running inference...")
            mask = segment_from_points(model, processor, inference_session, frame, device=DEVICE)
            print(f"Drawing Mask...")
            display = draw_mask_overlay(display, mask, color=(0, 255, 100), alpha=0.45)
            points = []
            labels = []

        # Draw points
        #draw_points(display, points, labels)

        # HUD
        num_fg = sum(1 for l in labels if l == 1)
        num_bg = sum(1 for l in labels if l == 0)
        hud = f"Points: {len(points)} (FG:{num_fg} BG:{num_bg}) | Device: {DEVICE} | 'r'=reset  'q'=quit"
        cv2.putText(display, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # ESC
            break
        elif key == ord("r"):
            points.clear()
            labels.clear()
            needs_update = True
            print("Reset points and mask.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()