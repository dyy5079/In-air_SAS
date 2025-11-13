import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import ndimage
from ImgProcessing import findCenter, read_h5
from sklearn.metrics import roc_curve, auc

dataListO = glob.glob('Z:/PSU PhD/Projects/In-air_SAS/outputs/chip_h5/t3e1_*.h5')
print(len(dataListO))
dataListQ = glob.glob('Z:/PSU PhD/Projects/In-air_SAS/outputs/chip_h5/t4e1_*.h5')
print(len(dataListQ))
DetCountO = []
DetCountQ = []
maxArea = 200
tailArea = 10
cutArea = 1.2
newMaskMM = 6
debug = False
MM = 3

# # Analyze False Positive Rate with 'Letter O' datasets
for threshold in np.arange(99, 75, -1):
    detCount = 0
    for datum in dataListO:
        chip = read_h5(datum)
        mask, center = findCenter(chip['chip'], debug=debug, threshold=threshold, cutArea=cutArea)
        # Zero all mask pixels within radius of n * (20.3/2) from center
        radius = cutArea * (20.3 / 2)
        height, width = mask.shape
        pixpercm_x = width / 50
        pixpercm_y = height / 50
        radiuspx_x = radius * pixpercm_x
        radiuspx_y = radius * pixpercm_y
        y_idx, x_idx = np.indices(mask.shape)
        dist = np.sqrt(((x_idx - center[0]) / radiuspx_x)**2 + ((y_idx - center[1]) / radiuspx_y)**2)
        mask[(dist <= 1)] = 0

        # The image size is 50cm x 50cm
        # extent = [0, 50, 0, 50]
        # plt.figure()
        # plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        # plt.title(f'Binary Mask after zeroing within radius {radius:.2f} px')
        # plt.xlabel('Along-track (cm)', fontsize=10)
        # plt.ylabel('Cross-track (cm)', fontsize=10)
        # plt.show()

        newMask = ndimage.binary_opening(mask, structure=np.ones((newMaskMM,newMaskMM)))
        if threshold == 55 and datum == dataListO[0]:
            plt.figure()
            extent = [0, 50, 0, 50]
            plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
            plt.title(f'Binary Mask after binary_opening({threshold}th Percentile Threshold)')
            plt.xlabel('Along-track (cm)', fontsize=10)
            plt.ylabel('Cross-track (cm)', fontsize=10)
            plt.show()

        # Remove connected components that exceed size threshold
        labeled_mask, num_features = ndimage.label(newMask)
        
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled_mask == i)
            if component_size > maxArea:
                newMask[labeled_mask == i] = 0  # Remove large components
        


        # Classify based on remaining mask area
        if newMask.sum() > tailArea:
            detCount += 1
    DetCountO.append(detCount)

    if threshold == 75:
        print(f"At threshold {threshold}, detCount: {detCount}")
    elif threshold == 99:
        print(f"At threshold {threshold}, detCount: {detCount}")


# Analyze True Positive Rate with 'Letter Q' datasets
for threshold in range(99, 75, -1):
    detCount = 0
    for datum in dataListQ:
        chip = read_h5(datum)
        mask, center = findCenter(chip['chip'], debug=debug, threshold=threshold, cutArea=cutArea)
        
        # Zero all mask pixels within radius of n * (20.3/2) from center
        radius = cutArea * (20.3 / 2)
        height, width = mask.shape
        pixpercm_x = width / 50
        pixpercm_y = height / 50
        radiuspx_x = radius * pixpercm_x
        radiuspx_y = radius * pixpercm_y
        y_idx, x_idx = np.indices(mask.shape)
        if center is None:
            print(f"Warning: Center not found for file {datum} at threshold {threshold}. Exiting")
            continue
        dist = np.sqrt(((x_idx - center[0]) / radiuspx_x)**2 + ((y_idx - center[1]) / radiuspx_y)**2)
        mask[(dist <= 1)] = 0

        # The image size is 50cm x 50cm
        # extent = [0, 50, 0, 50]
        # plt.figure()
        # plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        # plt.title(f'Binary Mask after zeroing within radius {radius:.2f} px')
        # plt.xlabel('Along-track (cm)', fontsize=10)
        # plt.ylabel('Cross-track (cm)', fontsize=10)
        # plt.show()

        newMask = ndimage.binary_opening(mask, structure=np.ones((newMaskMM,newMaskMM)))
        if threshold == 55 and datum == dataListQ[0]:
            plt.figure()
            extent = [0, 50, 0, 50]
            plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
            plt.title(f'Binary Mask after binary_opening({threshold}th Percentile Threshold)')
            plt.xlabel('Along-track (cm)', fontsize=10)
            plt.ylabel('Cross-track (cm)', fontsize=10)
            plt.show()


        # Remove connected components that exceed size threshold
        labeled_mask, num_features = ndimage.label(newMask)
        # Remove components that exceed threshold
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled_mask == i)
            if component_size > maxArea:
                newMask[labeled_mask == i] = 0  # Remove large components

        # Classify based on remaining mask area
        if newMask.sum() > tailArea:
            detCount += 1
    
    DetCountQ.append(detCount)

    if threshold == 75:
        print(f"At threshold {threshold}, detCount: {detCount}")
    elif threshold == 99:
        print(f"At threshold {threshold}, detCount: {detCount}")   
    
print("DetCountO:", DetCountO)
print("DetCountQ:", DetCountQ)

# Convert counts to rates for ROC calculation
total_O_files = len(dataListO)
total_Q_files = len(dataListQ)

# False Positive Rate = FP / (FP + TN) = DetCountO / total_O_files
FPR = np.array(DetCountO) / total_O_files
# True Positive Rate = TP / (TP + FN) = DetCountQ / total_Q_files  
TPR = np.array(DetCountQ) / total_Q_files

print(f"Total O files: {total_O_files}, Total Q files: {total_Q_files}")
print("False Positive Rates:", FPR)
print("True Positive Rates:", TPR)

# Calculate AUC using trapezoidal rule
roc_auc = np.trapezoid(TPR, FPR) if len(FPR) > 1 else 0

# Plot ROC curve
plt.figure()
plt.plot(FPR, TPR, color='darkorange', lw=2, marker='o', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()