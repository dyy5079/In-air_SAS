import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import ndimage
from ImgProcessing import findCenter, read_h5
from sklearn.metrics import roc_curve, auc

dataListO = glob.glob('Z:/PSU PhD/Projects/In-air_SAS/outputs/chip_h5/t3e2_*.h5')
print(len(dataListO))
dataListQ = glob.glob('Z:/PSU PhD/Projects/In-air_SAS/outputs/chip_h5/t4e2_*.h5')
print(len(dataListQ))
DetCountO = []
DetCountQ = []
maxArea = 200
cutArea = 1.3
newMaskMM = 6
debug = False
MM = 3
threshold = 95
# # Analyze False Positive Rate with 'Letter O' datasets

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

    # The image size is 75cm x 75cm
    # extent = [0, 75, 0, 75]
    # plt.figure()
    # plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
    # plt.title(f'Binary Mask after zeroing within radius {radius:.2f} px')
    # plt.xlabel('Along-track (cm)', fontsize=10)
    # plt.ylabel('Cross-track (cm)', fontsize=10)
    # plt.show()

    # newMask = ndimage.binary_opening(mask, structure=np.ones((newMaskMM,newMaskMM)))
    newMask = mask.copy()
    # Remove connected components that exceed size threshold
    labeled_mask, num_features = ndimage.label(newMask)
    
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_mask == i)
        if component_size > maxArea:
            newMask[labeled_mask == i] = 0  # Remove large components
    
    # Classify based on remaining mask area
    DetCountO.append(newMask.sum())


# Analyze True Positive Rate with 'Letter Q' datasets
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

    # The image size is 75cm x 75cm
    # extent = [0, 75, 0, 75]
    # plt.figure()
    # plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
    # plt.title(f'Binary Mask after zeroing within radius {radius:.2f} px')
    # plt.xlabel('Along-track (cm)', fontsize=10)
    # plt.ylabel('Cross-track (cm)', fontsize=10)
    # plt.show()

    newMask = ndimage.binary_opening(mask, structure=np.ones((newMaskMM,newMaskMM)))

    # Remove connected components that exceed size threshold
    labeled_mask, num_features = ndimage.label(newMask)
    # Remove components that exceed threshold
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_mask == i)
        if component_size > maxArea:
            newMask[labeled_mask == i] = 0  # Remove large components

    # Classify based on remaining mask area
    DetCountQ.append(newMask.sum())


# Plot histograms for DetCountO and DetCountQ
plt.figure(figsize=(12, 5))

# Subplot 1: DetCountO (Letter O datasets)
plt.subplot(1, 2, 1)
plt.hist(DetCountO, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Remaining Mask Area\n(Letter O - Expected Negatives)')
plt.xlabel('Remaining Mask Area (pixels)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Subplot 2: DetCountQ (Letter Q datasets)
plt.subplot(1, 2, 2)
plt.hist(DetCountQ, bins=20, alpha=0.7, color='red', edgecolor='black')
plt.title('Distribution of Remaining Mask Area\n(Letter Q - Expected Positives)')
plt.xlabel('Remaining Mask Area (pixels)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some basic statistics
print(f"DetCountO (Letter O) - Mean: {np.mean(DetCountO):.2f}, Std: {np.std(DetCountO):.2f}, Min: {np.min(DetCountO)}, Max: {np.max(DetCountO)}")
print(f"DetCountQ (Letter Q) - Mean: {np.mean(DetCountQ):.2f}, Std: {np.std(DetCountQ):.2f}, Min: {np.min(DetCountQ)}, Max: {np.max(DetCountQ)}")


