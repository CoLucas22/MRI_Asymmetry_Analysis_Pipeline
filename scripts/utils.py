import SimpleITK as sitk
import pandas as pd
import numpy as np
import cv2
from scipy.ndimage import rotate
from typing import Tuple

def divide_image(image):
    """Divide an image into two equal parts along the vertical axis.

    Args:
        image (numpy array): numpy array of the image

    Returns:
        numpy array: left/right splitted array
    """
    left_part = image[:, :image.shape[1] // 2]
    right_part = image[:, image.shape[1] // 2:]
    return right_part, left_part

def window_size_in_pixels(filename, window_height_cm=1, window_width_cm=6):
    """
    Calculates window size in pixels for a given DICOM image.
    
    :param filename: Path to DICOM file
    :param window_height_cm: Window height in centimeters
    :param window_width_cm: Window width in centimeters
    :return: Window size in pixels (height, width)
    """
    
    # Charger le fichier DICOM
    try:
        image = sitk.ReadImage(filename)
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier DICOM : {e}")
    
    # Obtenir l'espacement des pixels (en mm)
    spacing = image.GetSpacing()  # (spacing_x, spacing_y, spacing_z)
    
    if len(spacing) < 2:
        raise ValueError("L'espacement des pixels n'est pas valide pour une image 2D.")
    
    pixel_spacing_x, pixel_spacing_y = spacing[0], spacing[1]
    
    # Vérifier que les dimensions sont positives
    if window_height_cm <= 0 or window_width_cm <= 0:
        raise ValueError("Les dimensions de la fenêtre doivent être des valeurs positives.")
    
    # Convertir les dimensions de la fenêtre de cm à mm
    window_height_mm = window_height_cm * 10
    window_width_mm = window_width_cm * 10
    
    # Calculer la taille de la fenêtre en pixels
    window_height_pixels = window_height_mm / pixel_spacing_y
    window_width_pixels = window_width_mm / pixel_spacing_x
    
    #print(filename,spacing, (int(round(window_width_pixels)), int(round(window_height_pixels))))
    
    return (int(round(window_width_pixels)), int(round(window_height_pixels)))

def open_dcm(path):
    """Open a dicom file and return the image as a numpy array

    Args:
        path (str): Path to DICOM image
    """
    image = sitk.ReadImage(path, sitk.sitkFloat32)
       
    image = sitk.RescaleIntensity(image, outputMinimum=0, outputMaximum=255)

    image_array = sitk.GetArrayFromImage(image)
    image_array = np.squeeze(image_array)
    return image_array

def get_window_center(image_path, reference_file, images_folder, axis_points):
    """_summary_

    Args:
        image_path (str): Path to the DICOM image
        reference_file (str): File containing the reference points
        images_folder (str): Folder containing the DICOM images   
        axis_points (list): List of points on the symmetry axis    

    Returns:
        tuple: Coordinates of the window center (x, y)
    """
    df = pd.read_csv(reference_file)

    row = df[df['Run'] == "/".join(image_path.replace(images_folder,'').split('/')[-3:-1])]
    
    for point in axis_points:
        try:
            if point[1] == int(row['Y approximatif'].iloc[0]):
                return point
        except:
            print(row)
            print("/".join(image_path.replace(images_folder,'').split('/')[-3:-1]))

def calculate_correlation(args: Tuple[np.ndarray, int, int, int]) -> Tuple[float, int, int]:
    """Calcule la corrélation pour un angle et un offset donnés."""
    shifted_image, offset, angle, width = args

    rotated = rotate(shifted_image, angle, reshape=False)
    
    center = (width // 2) + offset
    size = min(center, width - center)
    
    left_side = rotated[:, center-size:center]
    right_side = cv2.flip(rotated[:, center:center+size], 1)
    
    if left_side.shape[1] != right_side.shape[1]:
        if left_side.shape[1] > right_side.shape[1]:
            left_side = left_side[:, left_side.shape[1] - right_side.shape[1]:] 
        else:
            right_side = right_side[:, right_side.shape[1] - left_side.shape[1]:] 

    max_corr = np.corrcoef(left_side.flatten(), right_side.flatten())[0, 1]
    return max_corr, angle, offset

def sliding_window_on_axis(image, axis_points:list, windowSize:tuple):
    """Slide a window over the image centered on the axis points

    Args:
        image (numpy array): _description_
        axis_points (list): _description_
        windowSize (tuple): _description_

    Yields:
        tuple: x and y coordinates of the center of the window and the window itself
    """
    half_width = windowSize[0] // 2
    half_height = windowSize[1] // 2

    for x, y in axis_points:
        x_start = int(max(0, x - half_width))
        x_end = int(min(image.shape[1], x + half_width))
        y_start = int(max(0, y - half_height))
        y_end = int(min(image.shape[0], y + half_height))

        
        window = image[y_start:y_end, x_start:x_end]
        yield (x, y, window)