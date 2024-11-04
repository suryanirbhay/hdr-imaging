from src.hdr_imaging import HDRImaging
import os
import logging
import cv2

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        hdr_processor = HDRImaging()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_paths = [
            'images/low.jpeg',
            'images/normal.jpeg',
            'images/high.jpeg'
            ]
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
        
        logger.info("Attempting to read exposure times from EXIF data...")
        actual_exposure_times = hdr_processor.get_exposure_times(image_paths)

        exposure_times = [1/4100, 1/910, 1/122]
        
        if actual_exposure_times and len(actual_exposure_times) == len(image_paths):
            logger.info("Using exposure times from EXIF data")
            exposure_times = actual_exposure_times
        else:
            logger.warning("Using default exposure times as EXIF data was not available")
            logger.info(f"Default exposure times: {exposure_times}")

        # Process images
        logger.info("Processing HDR images...")
        hdr, ldr = hdr_processor.process_hdr(
            image_paths,
            exposure_times,
            gamma=1.0,
            saturation=1.2
        )

        # Visualize results
        logger.info("Generating visualization...")
        hdr_processor.visualize_results(
            hdr_processor.read_images(image_paths),
            hdr,
            ldr
        )
        
        # Save results
        output_dir = os.path.join(current_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        hdr_path = os.path.join(output_dir, "hdr_result.hdr")
        ldr_path = os.path.join(output_dir, "tone_mapped_result.jpg")
        
        logger.info(f"Saving HDR image to: {hdr_path}")
        cv2.imwrite(hdr_path, hdr)
        
        logger.info(f"Saving tone-mapped image to: {ldr_path}")
        cv2.imwrite(ldr_path, ldr)
        
        logger.info("Processing completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()