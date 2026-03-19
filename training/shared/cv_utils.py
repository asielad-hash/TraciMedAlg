"""
Computer vision utilities — homography, mask warping, color conversion.
Extracted from trackInstruments/python/annotate_kf.py
"""

import cv2
import numpy as np


def compute_homography(prev_gray: np.ndarray, curr_gray: np.ndarray,
                        min_matches: int = 15) -> np.ndarray:
    """Compute homography between two grayscale frames using ORB + RANSAC.

    Returns: 3x3 homography matrix, or None if insufficient matches.
    """
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < min_matches:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H


def warp_mask(mask: np.ndarray, H: np.ndarray, size: tuple) -> np.ndarray:
    """Warp a binary mask using a homography matrix.

    Args:
        mask: HxW uint8 binary mask
        H: 3x3 homography matrix
        size: (width, height) of output

    Returns: warped binary mask
    """
    if H is None:
        return mask
    warped = cv2.warpPerspective(mask, H, size, flags=cv2.INTER_NEAREST)
    return (warped > 127).astype(np.uint8) * 255


def compute_cumulative_homographies(video_path: str, stride: int = 1,
                                      max_frames: int = None) -> list:
    """Pre-compute cumulative homographies from frame 0 to each frame.

    Returns: list of 3x3 matrices (or None where computation failed).
    H[i] transforms from frame 0 to frame i.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    homographies = [np.eye(3)]  # frame 0 → frame 0 = identity
    prev_gray = None

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if idx == 0:
            prev_gray = gray
            continue

        if idx % stride != 0:
            homographies.append(homographies[-1])  # carry forward
            prev_gray = gray
            continue

        H_inc = compute_homography(prev_gray, gray)
        if H_inc is not None:
            H_cum = H_inc @ homographies[-1]
            homographies.append(H_cum)
        else:
            homographies.append(homographies[-1])

        prev_gray = gray

    cap.release()
    return homographies


def anonymize_frame(frame_bgr: np.ndarray, mask: np.ndarray,
                     pixelate_factor: int = 16, blur_sigma: int = 15) -> np.ndarray:
    """Apply pixelation + Gaussian blur to masked regions (anonymization).

    Extracted from trackInstruments/python/anonymize_video.py
    """
    if mask is None or mask.sum() == 0:
        return frame_bgr

    # Dilate mask slightly for feathered edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)

    # Create feathered alpha
    alpha = cv2.GaussianBlur(dilated.astype(np.float32), (0, 0), blur_sigma)
    alpha = np.clip(alpha, 0, 1)

    # Pixelate
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (w // pixelate_factor, h // pixelate_factor),
                        interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Gaussian blur
    blurred = cv2.GaussianBlur(pixelated, (0, 0), blur_sigma)

    # Alpha blend
    alpha3 = alpha[:, :, None]
    result = (frame_bgr * (1 - alpha3) + blurred * alpha3).astype(np.uint8)
    return result
