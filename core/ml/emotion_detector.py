import cv2
import mediapipe as mp
import numpy as np

def detect_emotion(image_path):
    """
    Detects emotion from an image using Mediapipe Face Mesh landmarks.
    Returns:
        - processed_image_path (str): Path to the image with landmarks drawn.
        - emotion_label (str): Detected emotion (Feliz, Triste, Sorprendido, Neutral).
    """
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error al cargar la imagen"
            
        # Convert BGR to RGB
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        emotion_label = "Neutral"
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                # Analyze landmarks for emotion (Heuristic)
                landmarks = face_landmarks.landmark
                h, w, _ = image.shape
                
                # Key landmarks (indices based on standard facemesh)
                # Lips: Top(13), Bottom(14), Left(61), Right(291)
                # Eyebrows: Left_Top(105), Right_Top(334)
                
                mouth_top = landmarks[13].y
                mouth_bottom = landmarks[14].y
                mouth_left = landmarks[61].x
                mouth_right = landmarks[291].x
                
                mouth_open = mouth_bottom - mouth_top
                mouth_width = mouth_right - mouth_left
                
                # Normalize metrics
                aspect_ratio = mouth_open / (mouth_width + 1e-6)
                
                # Simple Heuristics
                if aspect_ratio > 0.3:
                    if mouth_width > 0.4: # Wide open mouth
                        emotion_label = "Sorprendido"
                    else:
                        emotion_label = "Feliz" # Open mouth smile
                elif landmarks[61].y < landmarks[13].y and landmarks[291].y < landmarks[13].y:
                    # Corners higher than center -> Smile
                     emotion_label = "Feliz"
                elif landmarks[61].y > landmarks[14].y + 0.02 and landmarks[291].y > landmarks[14].y + 0.02:
                     # Corners significantly lower than center -> Sad
                     emotion_label = "Triste"
                else:
                    emotion_label = "Neutral"

        # Save processed image
        output_path = image_path.replace('.', '_processed.')
        cv2.imwrite(output_path, image)
        
        return output_path, emotion_label
