import cv2 as cv
import numpy as np
import os

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# Loading exposure images into a list
img_fn = ['images/low.jpeg', 'images/normal.jpeg', 'images/high.jpeg']
img_list = [cv.imread(fn) for fn in img_fn]
exposure_times = np.array([1/4100, 1/910, 1/122], dtype=np.float32)

# Merge exposures to HDR image
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# Tonemap HDR image
tonemap1 = cv.createTonemap(gamma=1.5)  # Adjust gamma if needed
res_debevec = tonemap1.process(hdr_debevec.copy())

# Check for NaNs or Infs in tonemapped image
if np.isnan(res_debevec).any() or np.isinf(res_debevec).any():
    print("Warning: NaN or Inf values found in tonemapped HDR image")

# Normalize and convert to 8-bit
res_debevec_normalized = cv.normalize(res_debevec, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
res_debevec_8bit = np.clip(res_debevec_normalized * 255, 0, 255).astype('uint8')

# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# Normalize other HDR results
res_robertson_8bit = np.clip(cv.normalize(hdr_robertson, None, 0, 1, cv.NORM_MINMAX) * 255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')

# Save the results
cv.imwrite("outputs/ldr_debevec.jpg", res_debevec_8bit)
cv.imwrite("outputs/ldr_robertson.jpg", res_robertson_8bit)
cv.imwrite("outputs/fusion_mertens.jpg", res_mertens_8bit)