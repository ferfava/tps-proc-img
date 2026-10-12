import numpy as np
from docscan.filters import GrayscaleFilter, BWFilter
from dataclasses import dataclass
import cv2
from docscan import exceptions

def detect_document (image: np.ndarray) -> np.ndarray:
        image_shape = image.shape
        if image_shape[0] > image_shape[1]:
            larger_side = image_shape[0]
            larger_side_flag = 0
        else:
            larger_side = image_shape[1]
            larger_side_flag = 1
        

        scale = 600 / larger_side

        if larger_side_flag == 0:
            new_size_smaller_size = scale * image_shape[1]
        else:
            new_size_smaller_size = scale * image_shape[0]
                        
        reduced_image = cv2.resize(image, (int(600 if larger_side_flag == 1 else new_size_smaller_size ), int(600 if larger_side_flag == 0 else new_size_smaller_size )))
        gray_image = GrayscaleFilter().apply(reduced_image)
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blur_image, 50, 150)
        morphed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        image_contourns = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_contourns = sorted(image_contourns[0], key=cv2.contourArea, reverse=True)[:5]

        for c in image_contourns:
            aprox_image = cv2.approxPolyDP(c, 0.02 * float(cv2.arcLength(c, True)), True)
            if len (aprox_image) !=4: #Margen adicional para más flexibilidad
                aprox_image = cv2.approxPolyDP(c, 0.04 * float(cv2.arcLength(c, True)), True)
            
            if len(aprox_image) == 4 and cv2.isContourConvex(aprox_image) and cv2.contourArea(aprox_image) > (reduced_image.shape[0]*reduced_image.shape[1]* 0.15)  :
                
                pts = aprox_image.reshape(4,2)
                
                s = np.sum(pts, axis=1)     
                tl = pts[np.argmin(s)]
                br = pts[np.argmax(s)]
                d = np.diff(pts, axis=1)
                tr = pts[np.argmin(d)]
                bl = pts[np.argmax(d)]
                
                ordered_pts = np.array([tl, tr, br, bl], dtype=float) / scale
                return ordered_pts.tolist()

        raise exceptions.DocumentNotFound("No se encontró un documento en la imagen")
        
    
    