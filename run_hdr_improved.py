import cv2
from src.improved_hdr import ImprovedHDRImaging
hdr_processor = ImprovedHDRImaging()

# Define image paths and exposure times
image_paths = ['images/low.jpeg', 'images/normal.jpeg', 'images/high.jpeg']
exposure_times = [1/4100, 1/910 ,1/122]  # Adjust these to match your actual exposure times

# Process images
hdr, ldr = hdr_processor.process_hdr(
    image_paths,
    exposure_times,
    gamma=1.0,
    saturation=1.2
)

# Save results
cv2.imwrite('hdr_result.hdr', hdr)
cv2.imwrite('tone_mapped_result.jpg', ldr)

# Visualize results
images = [cv2.imread(path) for path in image_paths]
hdr_processor.visualize_results(images, hdr, ldr)