import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List, Tuple
import logging
from PIL import Image
from PIL.ExifTags import TAGS

class ImprovedHDRImaging:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def read_and_preprocess_images(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Read and preprocess images, returning both 8-bit and float versions
        """
        images_8bit = []
        images_float = []
        
        for path in image_paths:
            # Read image
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            
            # Store 8-bit version for response curve estimation
            images_8bit.append(img)
            
            # Create float version for HDR merge
            img_float = img.astype(np.float32) / 255.0
            # Apply bilateral filter to reduce noise while preserving edges
            img_float = cv2.bilateralFilter(img_float, 9, 75, 75)
            images_float.append(img_float)
        
        self.logger.info(f"Loaded {len(images_8bit)} images successfully")
        return images_8bit, images_float

    def improve_alignment(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhanced image alignment with robust feature matching
        """
        aligned_images = [images[0]]  # Reference image
        ref_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        
        for img in images[1:]:
            try:
                # Convert current image to grayscale
                curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect SIFT features
                sift = cv2.SIFT_create()
                keypoints1, descriptors1 = sift.detectAndCompute(ref_gray, None)
                keypoints2, descriptors2 = sift.detectAndCompute(curr_gray, None)
                
                if descriptors1 is None or descriptors2 is None:
                    self.logger.warning("No features detected, using original image")
                    aligned_images.append(img)
                    continue
                
                # Match features
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(descriptors1, descriptors2, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) < 4:
                    self.logger.warning("Not enough good matches found, using original image")
                    aligned_images.append(img)
                    continue
                
                # Get matched point pairs
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                if H is None:
                    self.logger.warning("Homography estimation failed, using original image")
                    aligned_images.append(img)
                    continue
                
                # Warp image
                aligned = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
                aligned_images.append(aligned)
                
            except Exception as e:
                self.logger.warning(f"Alignment failed: {str(e)}, using original image")
                aligned_images.append(img)
        
        return aligned_images

    def estimate_response_curve(self, images: List[np.ndarray], 
                              exposure_times: List[float],
                              samples_per_channel: int = 100) -> np.ndarray:
        """
        Improved response curve estimation with better sampling
        """
        # Ensure images are 8-bit
        images_8bit = [img if img.dtype == np.uint8 else (img * 255).astype(np.uint8) 
                      for img in images]
        
        # Sample pixels uniformly across image
        h, w = images_8bit[0].shape[:2]
        sample_points = np.random.choice(h * w, samples_per_channel, replace=False)
        
        # Create calibration object with smoothness term
        calibrateDebevec = cv2.createCalibrateDebevec(samples=samples_per_channel, lambda_=10.0, random=False)
        
        # Process images
        response_curve = calibrateDebevec.process(images_8bit, np.array(exposure_times, dtype=np.float32))
        
        return response_curve

    def merge_hdr(self, images: List[np.ndarray], 
                 exposure_times: List[float],
                 response_curve: np.ndarray = None) -> np.ndarray:
        """
        Enhanced HDR merging with weighted contribution
        """
        # Convert to 8-bit if necessary
        images_8bit = [img if img.dtype == np.uint8 else (img * 255).astype(np.uint8) 
                      for img in images]
        
        # Create merge object
        mergeDebevec = cv2.createMergeDebevec()
        
        # Merge images
        hdr = mergeDebevec.process(images_8bit, np.array(exposure_times, dtype=np.float32), response_curve)
        
        return hdr

    def advanced_tone_mapping(self, hdr_image: np.ndarray,
                            gamma: float = 1.0,
                            saturation: float = 1.2) -> np.ndarray:
        """
        Advanced tone mapping using Reinhard operator with enhancements
        """
        # Normalize HDR image if necessary
        if hdr_image.max() > 1.0:
            hdr_image = hdr_image / hdr_image.max()
        
        # Create Reinhard tonemap operator with enhanced parameters
        tonemap = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=0.0,  # Preserve overall intensity
            light_adapt=0.8,  # Local adaptation to light
            color_adapt=saturation  # Color saturation
        )
        
        # Apply tone mapping
        ldr = tonemap.process(hdr_image.copy())
        
        # Convert to 8-bit
        ldr_8bit = np.clip(ldr * 255, 0, 255).astype(np.uint8)
        
        # Enhance local contrast using CLAHE
        lab = cv2.cvtColor(ldr_8bit, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with adaptive parameters
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels back
        ldr_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Fine-tune contrast and brightness
        alpha = 1.1  # Contrast control
        beta = 5    # Brightness control
        ldr_final = cv2.convertScaleAbs(ldr_enhanced, alpha=alpha, beta=beta)
        
        return ldr_final

    def process_hdr(self, image_paths: List[str],
                   exposure_times: List[float],
                   gamma: float = 1.0,
                   saturation: float = 1.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete improved HDR pipeline
        """
        # Read and preprocess images
        images_8bit, images_float = self.read_and_preprocess_images(image_paths)
        
        # Align images (using 8-bit versions for feature detection)
        aligned_8bit = self.improve_alignment(images_8bit)
        
        # Estimate response curve
        response_curve = self.estimate_response_curve(aligned_8bit, exposure_times)
        
        # Merge to HDR
        hdr = self.merge_hdr(aligned_8bit, exposure_times, response_curve)
        
        # Tone map
        ldr = self.advanced_tone_mapping(hdr, gamma, saturation)
        
        self.logger.info(f"HDR image min/max: {hdr.min()}/{hdr.max()}")
        self.logger.info(f"LDR image min/max: {ldr.min()}/{ldr.max()}")
        
        return hdr, ldr

    def visualize_results(self, images: List[np.ndarray], 
                         hdr: np.ndarray, 
                         ldr: np.ndarray):
        """
        Create visualization of input images and results
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
        plt.imshow(cv2.cvtColor(np.clip(hdr/hdr.max(), 0, 1), cv2.COLOR_BGR2RGB))
        plt.title('HDR Image')
        plt.axis('off')
        
        # Show tone mapped result
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB))
        plt.title('Tone Mapped Result')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()