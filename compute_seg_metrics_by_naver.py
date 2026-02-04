# =============================================================================
# Description   : Compute segmentation metrics (IoU, Boundary IoU, Boundary F1)
# Author        : Deepixel
# Created       : 2025-05-26
# Python        : 3.10
# opencv-python : 410.0.84
# numpy         : 1.26.4
# =============================================================================

import cv2
import numpy as np
import sys
import os
import glob
from pathlib import Path


def boundary_precision(predicted_boundary: np.ndarray, gt_boundary: np.ndarray, thickness=15) -> float:
    """
    Calculate boundary precision.
    
    Parameters
    ----------
    predicted_boundary : np.ndarray of shape (H, W), dtype=np.uint8
        Predicted boundary mask.
        0 for background, >0 for boundary pixels.
        
    gt_boundary : np.ndarray of shape (H, W), dtype=np.uint8
        Ground truth boundary mask.
        0 for background, >0 for boundary pixels.
    
    thickness : int, optional
        Thickness of the boundary to be considered. Default is 15.
        
    Returns
    -------
    precision : float
        Boundary precision score, rounded to 2 decimal places.
    """
    if np.sum(predicted_boundary) == 0:
        return 0.0
    
    # Dilate ground truth boundary for tolerance
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_gt = cv2.dilate(gt_boundary, kernel, iterations=thickness)
    
    # Count true positives (predicted boundary points within dilated GT boundary)
    true_positives = np.sum(predicted_boundary & dilated_gt)
    
    # Calculate precision
    precision = true_positives / np.sum(predicted_boundary)
    
    return precision


def boundary_recall(predicted_boundary: np.ndarray, gt_boundary: np.ndarray, thickness=15) -> float:
    """
    Calculate boundary recall.
    
    Parameters
    ----------
    predicted_boundary : np.ndarray of shape (H, W), dtype=np.uint8
        Predicted boundary mask.
        0 for background, >0 for boundary pixels.
        
    gt_boundary : np.ndarray of shape (H, W), dtype=np.uint8
        Ground truth boundary mask.
        0 for background, >0 for boundary pixels.
    
    thickness : int, optional
        Thickness of the boundary to be considered. Default is 15.
        
    Returns
    -------
    recall : float
        Boundary recall score, rounded to 2 decimal places.
    """
    if np.sum(gt_boundary) == 0:
        return 0.0
    
    # Dilate predicted boundary for tolerance
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_pred = cv2.dilate(predicted_boundary, kernel, iterations=thickness)
    
    # Count true positives (GT boundary points within dilated predicted boundary)
    true_positives = np.sum(gt_boundary & dilated_pred)
    
    # Calculate recall
    recall = true_positives / np.sum(gt_boundary)
    
    return recall


def extract_boundary(mask: np.ndarray, thickness=15):
    """
    Extract the boundary of a binary mask using morphological operations.
    
    Parameters
    ----------
    mask : np.ndarray of shape (H, W), dtype=np.uint8
        mask from which to extract the boundary.
        0 for background, >0 for foreground.
    
    thickness : int, optional
        Thickness of the boundary to be extracted. Default is 15.
    
    Returns
    -------
    boundary_mask : np.ndarray of shape (H, W), dtype=np.uint8
        mask representing the boundary.
        0 for background, >0 for boundary pixels.
    """
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    boundary_mask = dilated_mask - eroded_mask
    
    return boundary_mask



def compute_iou(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) for binary masks.
    
    Parameters
    ----------
    true_mask : np.ndarray of shape (H, W), dtype=np.uint8
        Ground truth mask.
        0 for background, >0 for foreground.
        
    pred_mask : np.ndarray of shape (H, W), dtype=np.uint8
        Predicted mask.
        0 for background, >0 for foreground.
    
    Returns
    -------
    iou : float
        Intersection over Union score, rounded to 2 decimal places.
    """
    
    true_label = true_mask > 0
    true_class = pred_mask > 0
    
    intersection = np.sum(np.logical_and(true_label, true_class))
    union = np.sum(np.logical_or(true_label, true_class))

    iou = intersection / union if union > 0 else 0

    iou = round(iou * 100, 2)
    
    return iou


def compute_biou(true_mask: np.ndarray, pred_mask: np.ndarray, thickness=15) -> float:
    """
    Compute Boundary IoU (BIoU) for binary masks.
    
    Parameters
    ----------
    true_mask : np.ndarray of shape (H, W), dtype=np.uint8
        Ground truth mask.
        0 for background, >0 for foreground.
        
    pred_mask : np.ndarray of shape (H, W), dtype=np.uint8
        Predicted mask.
        0 for background, >0 for foreground.
        
    thickness : int, optional
        Thickness of the boundary to be extracted. Default is 15.
    
    Returns
    -------
    biou : float
        Boundary Intersection over Union score, rounded to 2 decimal places.
        
    """
    height, width = pred_mask.shape[:2]
    true_mask = cv2.resize(true_mask, (width, height))

    true_boundaries = extract_boundary(true_mask, thickness=thickness)
    pred_boundaries = extract_boundary(pred_mask, thickness=thickness)
    
    biou = compute_iou(true_boundaries, pred_boundaries)
    
    return biou


def compute_boundary_f1(true_mask: np.ndarray, pred_mask: np.ndarray, thickness=15, tolerance=2) -> float:
    """
    Compute Boundary F1 score for binary masks.
    
    Parameters
    ----------
    true_mask : np.ndarray of shape (H, W), dtype=np.uint8
        Ground truth mask.
        0 for background, >0 for foreground.
        
    pred_mask : np.ndarray of shape (H, W), dtype=np.uint8
        Predicted mask.
        0 for background, >0 for foreground.
        
    thickness : int, optional
        Thickness of the boundary to be extracted. Default is 15.
        
    tolerance : int, optional
        Tolerance for boundary matching. Default is 2.
        
    Returns
    -------
    f1 : float
        Boundary F1 score, rounded to 2 decimal places.
        
    """
    height, width = pred_mask.shape[:2]
    true_mask = cv2.resize(true_mask, (width, height))
    
    true_boundaries = extract_boundary(true_mask, thickness=thickness)
    pred_boundaries = extract_boundary(pred_mask, thickness=thickness)
    
    precision = boundary_precision(pred_boundaries, true_boundaries, thickness=tolerance)
    recall = boundary_recall(pred_boundaries, true_boundaries, thickness=tolerance)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1 = round(f1 * 100, 2)
    
    return f1


def check_alpha_channel_from_image(pred_img: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Check if image data has alpha channel and return the appropriate mask.
    
    Parameters
    ----------
    pred_img : np.ndarray
        Predicted image data (loaded with IMREAD_UNCHANGED).
        
    Returns
    -------
    pred_mask : np.ndarray
        Extracted mask (alpha channel if available, otherwise grayscale).
    has_alpha : bool
        Whether the image has alpha channel.
    """
    if pred_img is None:
        raise ValueError("Input image data is None")
    
    # 이미지 정보 출력
    # print(f"   Image shape: {pred_img.shape}")
    
    # 알파 채널이 있는지 확인
    if len(pred_img.shape) == 3 and pred_img.shape[2] == 4:  # RGBA
        pred_mask = pred_img[:, :, 3]  # 알파 채널 사용
        # print(f"   ✅ Alpha channel detected - Using alpha channel for mask")
        has_alpha = True
    elif len(pred_img.shape) == 3 and pred_img.shape[2] == 3:  # RGB (알파 채널 없음)
        # 그레이스케일로 변환하여 사용
        pred_mask = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        # print(f"   ⚠️  No alpha channel - Converting RGB to grayscale")
        has_alpha = False
    else:  # 이미 그레이스케일
        pred_mask = pred_img
        # print(f"   ⚠️  Grayscale image - Using as is")
        has_alpha = False
    
    # # 마스크 통계 정보 출력
    # unique_values = np.unique(pred_mask)
    # print(f"   Mask value range: {pred_mask.min()} - {pred_mask.max()}")
    # print(f"   Unique values count: {len(unique_values)}")
    # print(f"   Non-zero pixels: {np.sum(pred_mask > 0)} / {pred_mask.size} ({np.sum(pred_mask > 0)/pred_mask.size*100:.1f}%)")
    # print()
    
    return pred_mask, has_alpha


def compute_boundary_f1_with_alpha(true_mask: np.ndarray, pred_mask_path: str, target_size: tuple = (256, 256), thickness=15, tolerance=2) -> float:
    """
    Compute Boundary F1 score for binary masks using PNG alpha channel.
    
    Parameters
    ----------
    true_mask : np.ndarray of shape (H, W), dtype=np.uint8
        Ground truth mask.
        0 for background, >0 for foreground.
        
    pred_mask_path : str
        Path to the predicted PNG mask with alpha channel.
        
    target_size : tuple, optional
        Target size for resizing both masks. Default is (256, 256).
        If None, uses GT mask size.
        
    thickness : int, optional
        Thickness of the boundary to be extracted. Default is 15.
        
    tolerance : int, optional
        Tolerance for boundary matching. Default is 2.
        
    Returns
    -------
    f1 : float
        Boundary F1 score using alpha channel, rounded to 2 decimal places.
        
    """
    # 알파 채널 확인 및 마스크 추출
    pred_img = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)
    pred_mask, has_alpha = check_alpha_channel_from_image(pred_img)
    
    # 크기 통일 처리
    if target_size:
        true_mask = cv2.resize(true_mask, target_size, interpolation=cv2.INTER_LINEAR)
        pred_mask = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        # target_size가 None이면 GT 크기에 맞춤
        height, width = true_mask.shape[:2]
        pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_LINEAR)
    
    true_boundaries = extract_boundary(true_mask, thickness=thickness)
    pred_boundaries = extract_boundary(pred_mask, thickness=thickness)
    
    precision = boundary_precision(pred_boundaries, true_boundaries, thickness=tolerance)
    recall = boundary_recall(pred_boundaries, true_boundaries, thickness=tolerance)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1 = round(f1 * 100, 2)
    
    return f1


def compute_biou_with_alpha(true_mask: np.ndarray, pred_mask_path: str, target_size: tuple = (256, 256), thickness=15) -> float:
    """
    Compute Boundary IoU (BIoU) for binary masks using PNG alpha channel.
    
    Parameters
    ----------
    true_mask : np.ndarray of shape (H, W), dtype=np.uint8
        Ground truth mask.
        0 for background, >0 for foreground.
        
    pred_mask_path : str
        Path to the predicted PNG mask with alpha channel.
        
    target_size : tuple, optional
        Target size for resizing both masks. Default is (256, 256).
        If None, uses GT mask size.
        
    thickness : int, optional
        Thickness of the boundary to be extracted. Default is 15.
    
    Returns
    -------
    biou : float
        Boundary Intersection over Union score using alpha channel, rounded to 2 decimal places.
        
    """
    # 알파 채널 확인 및 마스크 추출
    pred_img = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)
    pred_mask, has_alpha = check_alpha_channel_from_image(pred_img)
    
    # 크기 통일 처리
    if target_size:
        true_mask = cv2.resize(true_mask, target_size, interpolation=cv2.INTER_LINEAR)
        pred_mask = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        # target_size가 None이면 GT 크기에 맞춤
        height, width = true_mask.shape[:2]
        pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    true_boundaries = extract_boundary(true_mask, thickness=thickness)
    pred_boundaries = extract_boundary(pred_mask, thickness=thickness)
    
    biou = compute_iou(true_boundaries, pred_boundaries)
    
    return biou


def compute_metrics(gt_paths: list[str], pred_mask_paths: list[str], use_alpha_for_boundary: bool = True) -> tuple[float, float, float]:
    """
    Compute metrics for the portrait segmentation.
    
    Parameters
    ----------
    gt_paths : list of str
        List of paths to the ground truth masks.
    
    pred_mask_paths : list of str
        List of paths to the predicted binary masks.
        
    use_alpha_for_boundary : bool, optional
        Whether to use alpha channel for boundary metrics (BIoU and Boundary F1) calculation. Default is True.
    
    Returns
    -------
        miou (float): Mean Intersection over Union (using grayscale).
        mean_bd_f1 (float): Mean Boundary F1 score.
        mean_biou (float): Mean Boundary IoU.
        
    """
    print("=" * 80)
    print("🚀 Starting Segmentation Metrics Computation")
    print(f"📊 Total images to process: {len(pred_mask_paths)}")
    print(f"🎯 Using alpha channel for boundary metrics: {use_alpha_for_boundary}")
    print("=" * 80)
    print()
    
    TARGET_SIZE = (256, 256)
    
    miou = 0
    mean_bd_f1 = 0
    mean_biou = 0
    alpha_count = 0
    cnt = 0
    
    for gt_path, pred_mask_path in zip(gt_paths, pred_mask_paths):
        # 파일 존재 여부 확인
        if not os.path.exists(gt_path):
            print(f"❌ Error: GT file not found: {gt_path}")
            continue
        if not os.path.exists(pred_mask_path):
            print(f"❌ Error: Prediction file not found: {pred_mask_path}")
            continue
            
        print(f"📁 Processing GT: {gt_path}")
        print(f"📁 Processing Pred: {pred_mask_path}")
        
        # 1. GT 읽기 및 리사이즈 (그레이스케일)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"❌ Error: Cannot read GT image: {gt_path}")
            continue
        resized_gt = cv2.resize(gt, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # 2. Pred 읽기 및 알파 채널 확인
        pred_img = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)
        if pred_img is None:
            print(f"❌ Error: Cannot read prediction image: {pred_mask_path}")
            continue
            
        # 3. 알파 채널 처리
        if len(pred_img.shape) == 3 and pred_img.shape[2] == 4:  # RGBA
            # pred_mask = pred_img[:, :, 3]  # 알파 채널 사용

            # ------------------- ➕ 추가 1----------------------------
            # rgb 이미지 추출 및 rgb 이미지를 grayscale로 변환
            sample = cv2.cvtColor(pred_img[:, :, :3], cv2.COLOR_BGR2GRAY)
            # foreground (사람)만 픽셀을 1로 만들고, 배경은 0으로 변환
            sample = np.where(sample > 0, 1, 0)
            # unsigned integer 8bit로 변환
            pseudo_mask = sample.astype(np.uint8)

            # boundary = 255-pred_img[:, :, 3]
            # boundary = np.where(boundary > 0, 200, 0)
            # boundary = boundary.astype(np.uint8)
            
            # pred_mask = pseudo_mask - boundary
            pred_mask = pseudo_mask
            # ------------------------------------------------------
            
            alpha_count += 1
            print(f"   ✅ Using alpha channel for metrics")
        else:
            # 알파 채널 없음 - 그레이스케일로 변환
            if len(pred_img.shape) == 3:
                pred_mask = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
            else:
                pred_mask = pred_img
            print(f"   ⚠️  No alpha channel - using grayscale")
        
        # 4. Pred 마스크 리사이즈
        resized_pred_mask = cv2.resize(pred_mask, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # -------------------➕ 추가 2----------------------------
        # 이 코드는 메트릭 연산의 안정성을 위해 추가된 코드입니다.
        
        # foreground (사람)만 픽셀을 1로 만들고, 배경은 0으로 변환
        resized_gt = np.where(resized_gt > 0, 1, 0)
        # unsigned integer 8bit로 변환
        resized_gt = resized_gt.astype(np.uint8)
        # ------------------------------------------------------
        
        # 5. 메트릭 계산
        # IoU 계산
        iou = compute_iou(resized_gt, resized_pred_mask)
        miou += iou
        
        # BIoU 계산
        true_boundaries = extract_boundary(resized_gt, thickness=15)
        pred_boundaries = extract_boundary(resized_pred_mask, thickness=15)
        biou = compute_iou(true_boundaries, pred_boundaries)
        mean_biou += biou
        
        # Boundary F1 계산
        precision = boundary_precision(pred_boundaries, true_boundaries, thickness=2)
        recall = boundary_recall(pred_boundaries, true_boundaries, thickness=2)
        bd_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        bd_f1 = round(bd_f1 * 100, 2)
        mean_bd_f1 += bd_f1

        cnt += 1
        print(f"✅ Processed {cnt}/{len(pred_mask_paths)} images")
        print("-" * 50)
    
    if cnt == 0:
        print("❌ Error: No images were successfully processed!")
        return 0.0, 0.0, 0.0
    
    miou = miou / cnt
    mean_bd_f1 = mean_bd_f1 / cnt
    mean_biou = mean_biou / cnt
    
    # 최종 결과 출력
    print()
    print("=" * 80)
    print("📈 FINAL RESULTS")
    print("=" * 80)
    print(f"🔍 Alpha channel statistics:")
    print(f"   - Images with alpha channel: {alpha_count}/{cnt} ({alpha_count/cnt*100:.1f}%)")
    print(f"   - Images without alpha channel: {cnt-alpha_count}/{cnt} ({(cnt-alpha_count)/cnt*100:.1f}%)")
    print()
    
    print(f"📊 Metrics Summary:")
    print(f"   - Mean IoU: {miou:.2f}%")
    print(f"   - Mean Boundary F1: {mean_bd_f1:.2f}%")
    print(f"   - Mean Boundary IoU: {mean_biou:.2f}%")
    print("=" * 80)
    
    return miou, mean_bd_f1, mean_biou


def get_image_files(path: str) -> list[str]:
    """
    Get list of image files from a path (file or directory).
    
    Parameters
    ----------
    path : str
        Path to file or directory.
        
    Returns
    -------
    image_files : list[str]
        List of image file paths.
    """
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        # 지원하는 이미지 확장자
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        for ext in extensions:
            pattern = os.path.join(path, ext)
            image_files.extend(glob.glob(pattern))
            # 대소문자 구분 없이 검색
            pattern_upper = os.path.join(path, ext.upper())
            image_files.extend(glob.glob(pattern_upper))
        
        # 중복 제거 및 정렬
        image_files = sorted(list(set(image_files)))
        return image_files
    else:
        return []


def match_image_pairs(gt_path: str, pred_path: str) -> tuple[list[str], list[str]]:
    """
    Match GT and prediction image pairs from given paths by order (not by filename).
    
    Parameters
    ----------
    gt_path : str
        Path to GT file or directory.
    pred_path : str
        Path to prediction file or directory.
        
    Returns
    -------
    gt_files : list[str]
        List of matched GT file paths.
    pred_files : list[str]
        List of matched prediction file paths.
    """
    gt_files = get_image_files(gt_path)
    pred_files = get_image_files(pred_path)
    
    if not gt_files:
        print(f"❌ No image files found in GT path: {gt_path}")
        return [], []
    
    if not pred_files:
        print(f"❌ No image files found in prediction path: {pred_path}")
        return [], []
    
    print(f"📁 GT folder contains {len(gt_files)} images")
    print(f"📁 Prediction folder contains {len(pred_files)} images")
    
    # 순서대로 매칭 (파일명 비교 없음)
    min_count = min(len(gt_files), len(pred_files))
    
    if len(gt_files) != len(pred_files):
        print(f"⚠️  Warning: Different number of files in folders!")
        print(f"   GT: {len(gt_files)} files, Pred: {len(pred_files)} files")
        print(f"   Will process first {min_count} pairs")
    
    matched_gt = gt_files[:min_count]
    matched_pred = pred_files[:min_count]
    
    print(f"\n📊 Matching pairs by order:")
    for i, (gt_file, pred_file) in enumerate(zip(matched_gt, matched_pred), 1):
        print(f"   {i:3d}. {Path(gt_file).name} ↔ {Path(pred_file).name}")
        if i >= 5 and len(matched_gt) > 10:  # 처음 5개와 마지막 5개만 표시
            if i == 5:
                print(f"   ... (showing first 5 and last 5 of {len(matched_gt)} pairs)")
            if i < len(matched_gt) - 4:
                continue
    
    return matched_gt, matched_pred


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compute_seg_metrics_org.py <gt_path> <pred_path>")
        print("  <gt_path>   : Path to GT file or directory")
        print("  <pred_path> : Path to prediction file or directory")
        print()
        print("Examples:")
        print("  python compute_seg_metrics_org.py gt.png pred.png")
        print("  python compute_seg_metrics_org.py /path/to/gt_folder /path/to/pred_folder")
        print("  python compute_seg_metrics_org.py gt_folder pred_folder")
        sys.exit(1)
    
    gt_input = sys.argv[1]
    pred_input = sys.argv[2]
    
    print("=" * 80)
    print("🔍 ANALYZING INPUT PATHS")
    print("=" * 80)
    print(f"GT Input: {gt_input}")
    print(f"Prediction Input: {pred_input}")
    print()
    
    # 입력 경로 타입 확인
    gt_is_dir = os.path.isdir(gt_input)
    pred_is_dir = os.path.isdir(pred_input)
    
    print(f"GT Input Type: {'Directory' if gt_is_dir else 'File'}")
    print(f"Prediction Input Type: {'Directory' if pred_is_dir else 'File'}")
    print()
    
    # 이미지 파일 매칭
    gt_paths, pred_mask_paths = match_image_pairs(gt_input, pred_input)
    
    if not gt_paths or not pred_mask_paths:
        print("❌ No matching image pairs found!")
        sys.exit(1)
    
    print(f"\n📊 Found {len(gt_paths)} matching image pairs")
    print("=" * 80)
    
    # 알파 채널을 경계선 메트릭에 사용
    miou, mean_bd_f1, mean_biou = compute_metrics(gt_paths, pred_mask_paths, use_alpha_for_boundary=True)
    
    # 기존 방식과 비교하고 싶다면
    # miou_old, mean_bd_f1_old, mean_biou_old = compute_metrics(gt_paths, pred_mask_paths, use_alpha_for_boundary=False)
    # print(f"Mean Boundary F1 (without Alpha): {mean_bd_f1_old:.2f}%")
    # print(f"Mean Boundary IoU (without Alpha): {mean_biou_old:.2f}%")