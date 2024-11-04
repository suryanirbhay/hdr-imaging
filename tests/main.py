from ..src.hdr_imaging import HDRImaging

hdr_processor = HDRImaging()

# Define input images and exposure times
image_paths = [
    'images/exposure1.jpg',
    'images/exposure2.jpg',
    'images/exposure3.jpg'
]
exposure_times = [1/30.0, 1/8.0, 1/2.0]  # Exposure times in seconds

# Process images
hdr, ldr = hdr_processor.process_hdr(
    image_paths,
    exposure_times,
    gamma=1.0,
    saturation=1.2
)

# Visualize results
hdr_processor.visualize_results(
    hdr_processor.read_images(image_paths),
    hdr,
    ldr
)