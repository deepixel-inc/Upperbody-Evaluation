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



def compute_metrics(gt_paths: list[str], pred_mask_paths: list[str]) -> tuple[float, float, float]:
    """
    Compute metrics for the portrait segmentation.
    
    Parameters
    ----------
    gt_paths : list of str
        List of paths to the ground truth masks.
    
    pred_mask_paths : list of str
        List of paths to the predicted binary masks.
    
    Returns
    -------
        miou (float): Mean Intersection over Union.
        mean_bd_f1 (float): Mean Boundary F1 score.
        mean_biou (float): Mean Boundary IoU.
        
    """
    miou = 0
    mean_bd_f1 = 0
    mean_biou = 0

    cnt = 0
    for gt_path, pred_mask_path in zip(gt_paths, pred_mask_paths):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)
        
        if pred_mask is None:
            print(f"❌ Error: Cannot read prediction image: {pred_mask_path}")
            continue
        
        # 알파 채널 처리
        if len(pred_mask.shape) == 3 and pred_mask.shape[2] == 4:  # BGRA
            sample = cv2.cvtColor(pred_mask[:, :, :3], cv2.COLOR_BGR2GRAY)
            sample = np.where(sample > 0, 1, 0)
            pred_mask = sample.astype(np.uint8)

        else:
            # 알파 채널 없음 - 그레이스케일로 변환
            if len(pred_mask.shape) == 3:
                pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
            else:
                pred_mask = pred_mask
                
            pred_mask = np.where(pred_mask > 0, 1, 0)
            pred_mask = pred_mask.astype(np.uint8)
            print(f"   ⚠️  No alpha channel - using grayscale")
        
        gt = np.where(gt > 0, 1, 0)
        gt = gt.astype(np.uint8)
        
        resized_gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_LINEAR)
        resized_pred_mask = cv2.resize(pred_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        iou = compute_iou(resized_gt, resized_pred_mask)
        miou += iou
        
        biou = compute_biou(resized_gt, resized_pred_mask, thickness=15)
        mean_biou += biou
        
        bd_f1 = compute_boundary_f1(resized_gt, resized_pred_mask, thickness=15, tolerance=2)
        mean_bd_f1 += bd_f1

        cnt += 1
    
    miou = miou / cnt
    mean_bd_f1 = mean_bd_f1 / cnt
    mean_biou = mean_biou / cnt
    
    return miou, mean_bd_f1, mean_biou


if __name__ == "__main__":
    # Example usage
    gt_paths = ["path/to/gt1.png", "path/to/gt2.png"]
    pred_mask_paths = ["path/to/pred1.png", "path/to/pred2.png"]
    
    miou, mean_bd_f1, mean_biou = compute_metrics(gt_paths, pred_mask_paths)
    
    print(f"Mean IoU: {miou:.2f}%")
    print(f"Mean Boundary F1: {mean_bd_f1:.2f}%")
    print(f"Mean Boundary IoU: {mean_biou:.2f}%")