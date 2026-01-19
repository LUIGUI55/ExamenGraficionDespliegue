import cv2
import numpy as np

def detect_tumor(image_path):
    """
    Simulates tumor detection.
    Since real trained weights are missing, this function will:
    1. Load the image.
    2. Identify bright regions (common in some MRI sequences for tumors).
    3. Generate a heatmap overlay.
    4. Predict 'Tumor' if a significant bright region is found.
    
    Returns:
        - processed_image_path (str): Path to image with heatmap.
        - label (str): 'Tumor Detected' or 'No Tumor Detected'.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error loading image"
        
        # Preprocessing mainly for visualization
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Simple heuristic: Tumors often hyperintense (bright)
        # We look for the brightest connected component
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
        
        # Create a mask for high intensity
        thresh_val = max(200, maxVal * 0.8) # Dynamic threshold
        _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tumor_detected = False
        if contours:
            # Find largest contour
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100: # Minimum area to consider
                tumor_detected = True
                
                # Draw contour on original image
                cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
                
                # Create Heatmap
                heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
                
                # Overlay heatmap
                image = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        label = "Tumor Detectado" if tumor_detected else "No se detectó tumor"
        
        # Save output
        output_path = image_path.replace('.', '_tumor_processed.')
        cv2.imwrite(output_path, image)
        
        return output_path, label

    except Exception as e:
        print(f"Error in tumor detection: {e}")
        return image_path, "Error en procesamiento"
