import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List, Tuple
import logging
from PIL import Image
from PIL.ExifTags import TAGS
import os

class HDRImaging:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def read_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Read multiple images and return as list of numpy arrays
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            List of images as numpy arrays
        """
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            images.append(img)
        
        self.logger.info(f"Loaded {len(images)} images successfully")
        return images
    
    def get_exposure_times(self, image_paths):
        """
        Extract exposure times from image EXIF data
    
        Args:
        image_paths: List of paths to images
        
        Returns:
        List of exposure times in seconds
        """
        exposure_times = []
    
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    exif = img._getexif()
                    if exif is not None:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag == 'ExposureTime':
                                if isinstance(value, tuple):
                                    exposure_time = value[0] / value[1]  # Convert rational to float
                                else:
                                    exposure_time = float(value)
                                exposure_times.append(exposure_time)
                                print(f"{os.path.basename(path)}: {exposure_time:.3f} seconds")
                                break
                        else:
                            print(f"No exposure time found for {os.path.basename(path)}")
                    else:
                        print(f"No EXIF data found for {os.path.basename(path)}")
            except Exception as e:
                print(f"Error reading {os.path.basename(path)}: {str(e)}")
    
        return exposure_times
    
    def align_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Align multiple images using ECC algorithm
        
        Args:
            images: List of input images
            
        Returns:
            List of aligned images
        """
        aligned_images = [images[0]]  # Reference image
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        
        for img in images[1:]:
            try:
                (cc, warp_matrix) = cv2.findTransformECC(
                    cv2.cvtColor(aligned_images[0], cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    warp_matrix,
                    warp_mode,
                    criteria
                )
                aligned = cv2.warpPerspective(img, warp_matrix, 
                                           (img.shape[1], img.shape[0]),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_images.append(aligned)
            except:
                self.logger.warning("Failed to align image, using original")
                aligned_images.append(img)
                
        return aligned_images
    
    def estimate_response_curve(self, images: List[np.ndarray], 
                              exposure_times: List[float]) -> np.ndarray:
        """
        Estimate camera response curve using Debevec's method
        
        Args:
            images: List of aligned images
            exposure_times: List of exposure times for each image
            
        Returns:
            Response curve for each color channel
        """
        calibrateDebevec = cv2.createCalibrateDebevec()
        response_curve = calibrateDebevec.process(images, np.array(exposure_times))
        return response_curve
    
    def merge_hdr(self, images: List[np.ndarray], 
                 exposure_times: List[float]) -> np.ndarray:
        """
        Merge multiple exposure images into HDR image
        
        Args:
            images: List of aligned images
            exposure_times: List of exposure times
            
        Returns:
            HDR image as numpy array
        """
        exposure_times = np.array(exposure_times, dtype=np.float32)
        response_curve = self.estimate_response_curve(images, exposure_times)
        mergeDebevec = cv2.createMergeDebevec()
        hdr = mergeDebevec.process(images, exposure_times, response_curve)
        return hdr
    
    def tone_map(self, hdr_image: np.ndarray, 
                 gamma: float = 1.0, 
                 saturation: float = 1.0) -> np.ndarray:
        """
        Apply tone mapping to HDR image
        
        Args:
            hdr_image: Input HDR image
            gamma: Gamma correction value
            saturation: Color saturation adjustment
            
        Returns:
            Tone mapped LDR image
        """
        tonemap = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=0.0,
            light_adapt=0.8,
            color_adapt=saturation
        )
        ldr = tonemap.process(hdr_image)
        ldr = np.clip(ldr * 255, 0, 255).astype('uint8')
        return ldr
    
    def process_hdr(self, image_paths: List[str], 
                   exposure_times: List[float],
                   gamma: float = 1.0,
                   saturation: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete HDR pipeline from input images to tone mapped result
        
        Args:
            image_paths: List of input image paths
            exposure_times: List of exposure times
            gamma: Gamma correction for tone mapping
            saturation: Color saturation for tone mapping
            
        Returns:
            Tuple of (HDR image, tone mapped LDR image)
        """
        # Read and align images
        images = self.read_images(image_paths)
        aligned_images = self.align_images(images)
        
        # Create HDR image
        hdr = self.merge_hdr(aligned_images, exposure_times)
        
        # Tone map result
        ldr = self.tone_map(hdr, gamma, saturation)
        
        return hdr, ldr
    
    def visualize_results(self, images: List[np.ndarray], 
                         hdr: np.ndarray, 
                         ldr: np.ndarray):
        """
        Create visualization of input images and results
        
        Args:
            images: List of input images
            hdr: HDR result
            ldr: Tone mapped result
        """
        fig = plt.figure(figsize=(15, 8))
        
        # Show input images
        for i, img in enumerate(images):
            plt.subplot(2, 3, i+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Exposure {i+1}')
            plt.axis('off')
        
        # Show HDR result
        plt.subplot(2, 3, 4)
        plt.imshow(np.clip(hdr, 0, 1))
        plt.title('HDR Image')
        plt.axis('off')
        
        # Show tone mapped result
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB))
        plt.title('Tone Mapped Result')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()