import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import ndimage
from ImgProcessing import findCenter, read_h5
from sklearn.metrics import roc_curve, auc

dataListO = glob.glob('Z:/PSU PhD/Projects/In-air_SAS/outputs/chip_h5/t3e2_0*.h5')
print(len(dataListO))
dataListQ = glob.glob('Z:/PSU PhD/Projects/In-air_SAS/outputs/chip_h5/t4e2_0*.h5')
print(len(dataListQ))
DetCountO = []
DetCountQ = []


# # Analyze False Positive Rate with 'Letter O' datasets
for threshold in range(99, 74, -1):
    detCount = 0
    for datum in dataListO:
        chip = read_h5(datum)
        mask, center = findCenter(chip['chip'], debug=False, threshold=threshold)

        # Zero all mask pixels within radius of 1.25 * (20.3/2) from center
        radius = 1.25 * (20.3 / 2)
        height, width = mask.shape
        pixpercm_x = width / 50
        pixpercm_y = height / 50
        radiuspx_x = radius * pixpercm_x
        radiuspx_y = radius * pixpercm_y
        y_idx, x_idx = np.indices(mask.shape)
        dist = np.sqrt(((x_idx - center[0]) / radiuspx_x)**2 + ((y_idx - center[1]) / radiuspx_y)**2)
        mask[(dist <= 1)] = 0

        # The image size is 50cm x 50cm
        extent = [0, 50, 0, 50]
        # plt.figure()
        # plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        # plt.title(f'Binary Mask after zeroing within radius {radius:.2f} px')
        # plt.xlabel('Along-track (cm)', fontsize=10)
        # plt.ylabel('Cross-track (cm)', fontsize=10)
        # plt.show()

        newMask = ndimage.binary_opening(mask, structure=np.ones((8,8)))
        # plt.figure()
        # plt.imshow(newMask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        # plt.title(f'Binary Mask after binary_opening({threshold}th Percentile Threshold)')
        # plt.xlabel('Along-track (cm)', fontsize=10)
        # plt.ylabel('Cross-track (cm)', fontsize=10)
        # plt.show()

        # Classify based on remaining mask area
        if newMask.sum() > 10:
            detCount += 1
    DetCountO.append(detCount)

    if threshold == 50:
        print(f"At threshold {threshold}, detCount: {detCount}")
    elif threshold == 99:
        print(f"At threshold {threshold}, detCount: {detCount}")


# Analyze True Positive Rate with 'Letter Q' datasets
for threshold in range(99, 74, -1):
    detCount = 0
    for datum in dataListQ:
        chip = read_h5(datum)
        mask, center = findCenter(chip['chip'], debug=False, threshold=threshold)
        
        # Zero all mask pixels within radius of 1.25 * (20.3/2) from center
        radius = 1.25 * (20.3 / 2)
        height, width = mask.shape
        pixpercm_x = width / 50
        pixpercm_y = height / 50
        radiuspx_x = radius * pixpercm_x
        radiuspx_y = radius * pixpercm_y
        y_idx, x_idx = np.indices(mask.shape)
        dist = np.sqrt(((x_idx - center[0]) / radiuspx_x)**2 + ((y_idx - center[1]) / radiuspx_y)**2)
        mask[(dist <= 1)] = 0

        # The image size is 50cm x 50cm
        extent = [0, 50, 0, 50]
        plt.figure()
        plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        plt.title(f'Binary Mask after zeroing within radius {radius:.2f} px')
        plt.xlabel('Along-track (cm)', fontsize=10)
        plt.ylabel('Cross-track (cm)', fontsize=10)
        plt.show()

        newMask = ndimage.binary_opening(mask, structure=np.ones((8,8)))
        # plt.figure()
        # plt.imshow(newMask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        # plt.title(f'Binary Mask after binary_opening({threshold}th Percentile Threshold)')
        # plt.xlabel('Along-track (cm)', fontsize=10)
        # plt.ylabel('Cross-track (cm)', fontsize=10)
        # plt.show()

        # Classify based on remaining mask area
        if newMask.sum() > 10:
            detCount += 1
    
    DetCountQ.append(detCount)

    if threshold == 50:
        print(f"At threshold {threshold}, detCount: {detCount}")
    elif threshold == 99:
        print(f"At threshold {threshold}, detCount: {detCount}")   
    
print("DetCountO:", DetCountO)
print("DetCountQ:", DetCountQ)

# After your classification loop, assuming you have:
# DetCountQ: array of detection scores for 'Letter Q' files
# DetCountO: array of detection scores for 'Letter O' files

# Combine arrays and create labels
scores = np.concatenate([DetCountQ, DetCountO])
labels = np.concatenate([np.ones(len(DetCountQ)), np.zeros(len(DetCountO))])

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


