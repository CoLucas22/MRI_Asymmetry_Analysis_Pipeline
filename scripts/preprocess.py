import itertools
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import pywt
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from utils import open_dcm, get_window_center, calculate_correlation, window_size_in_pixels



def test_simple_find_symmetry_axis(image: Optional[np.ndarray] = None, 
                               lissage: bool = False, angle=None) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """
    Trouve l'axe de symétrie de l'image de manière simple, en utilisant le multiprocessing.
    
    Args:
        image (np.ndarray, optional): Image numpy
        lissage (bool, optional): Appliquer un flou gaussien
    
    Returns:
        Tuple contenant l'image avec l'axe, l'angle, l'offset et l'image alignée
    """
    if image is None:
        raise ValueError('Image invalide')

    image_blur = gaussian_filter(image, 2) if lissage else image
    height, width = image_blur.shape
    
    tasks = []
    range_edge = image_blur.shape[1] // 10
    for offset in range(-range_edge, range_edge):
        tasks.append((image_blur, offset, angle, width))

    with Pool(cpu_count()) as p:
        results = p.map(calculate_correlation, tasks)
    
    best_result = type('BestResult', (), {
        'angle': 0, 
        'offset': 0, 
        'value': float('-inf')
    })()
    
    for value, angle, offset in results:
        if value > best_result.value:
            best_result.value = value
            best_result.angle = angle
            best_result.offset = offset
    
    aligned_image = rotate(image_blur, best_result.angle, reshape=False)
    aligned_image_with_axis = aligned_image.copy()
    center = (width // 2) + best_result.offset
    cv2.line(aligned_image_with_axis, (center, 0), (center, height), (255, 0, 0), 1)
    return aligned_image_with_axis, best_result.angle, best_result.offset, aligned_image, best_result.value

def symmetry_by_k(image_path: str, 
                   k: int = 5, 
                   lissage: bool = False) -> Tuple[Tuple[float, int], np.ndarray]:
    """
    Trouve l'axe de symétrie en divisant l'image en k tranches.
    
    Args:
        image_path (str): Chemin de l'image
        k (int, optional): Nombre de tranches. Défaut à 5
        lissage (bool, optional): Appliquer un flou gaussien
    
    Returns:
        Tuple contenant l'angle moyen et l'axe x, et l'image alignée
    """
    best_results = {"angle": 0, "offset": 0, "value": float('-inf')}
    symmetry_axes = []
    for angle in range(-3, 4):
        image = open_dcm(image_path)
        height, width = image.shape
        slice_height = height // k
        
        
        
        for i in range(k):
            start = i * slice_height
            end = start + slice_height if i < k - 1 else height
            
            slice_image = image[start:end, :]
            _, angle, offset, _, correlation = test_simple_find_symmetry_axis(image=slice_image, angle=angle, lissage=lissage)
            
            symmetry_axes.append(([angle, width//2 + offset]))
            mean_axis = np.median(symmetry_axes, axis=0)
        
            if correlation > best_results["value"]:

                best_results["value"] = correlation
                best_results["angle"] = angle
                best_results["offset"] = offset
        
        # Calcul de l'axe médian
    image_no_axis = rotate(image, best_results['angle'], reshape=False, order=0)
        
    
    return mean_axis, image_no_axis, best_results["angle"]

def extract_window(image_path, image=None, image_folder='./data_example/', wavelets = False, metadata_file = "Runs_dataset.csv"):
    """Extract a window from a DICOM image centered on the specified coordinates.

    Args:
        image_path (str): Path to the DICOM image
        image_preprocessed (np.ndarray): Preprocessed image
    Returns:
        numpy array: Extracted window
    """
    #print(parameters)
    #check si image_folder est deja dans image_path
    if wavelets == True:
        if image_folder not in image_path:
            image_path = os.path.join(image_folder,image_path) 

        (mean_angle, x_axis),axis_image,_ = symmetry_by_k(image_path, k=8)
        symmetry_axis_points = [(x_axis, i) for i in range(axis_image.shape[0])]
        

    else:
        if image_folder not in image_path:
            image_path = os.path.join(image_folder,image_path) 
        if image is None:
            (mean_angle, x_axis), image, _ = symmetry_by_k(image_path, k=8)
        else:
            (mean_angle, x_axis), _,_ = symmetry_by_k(image_path, k=8)


        symmetry_axis_points = [(x_axis, i) for i in range(image.shape[0])]
    

    window_center = get_window_center(image_path, reference_file=metadata_file, images_folder=image_folder, axis_points=symmetry_axis_points)   
        
    window_size_pixel = window_size_in_pixels(image_path, window_height_cm= 1, window_width_cm=7)
    
    if wavelets == True:
        window_center = (window_center[0]//2, window_center[1]//2)
        window_size_pixel = (window_size_pixel[0]//2, window_size_pixel[1]//2)
    

    x_center, y_center = window_center
    x_start = max(0, int(x_center - window_size_pixel[0] // 2))
    x_end = min(image.shape[1], int(x_center + window_size_pixel[0] // 2))
    y_start = max(0, int(y_center - window_size_pixel[1] // 2))
    y_end = min(image.shape[0], int(y_center + window_size_pixel[1] // 2))
    return image[y_start:y_end, x_start:x_end]

def preprocessing_image(image_path:str=None, image:np.ndarray=None, gaussian_smoothing:bool=None, 
                        wavelets:bool=None, extract_window_lesion:bool=False):
    """Preprocesses the input image using various filters and techniques.

    Args:
        image (np.ndarray): The input image to be preprocessed.
        gaussian_filter (bool): Apply Gaussian filter to smooth the image
        rescale_intensity (bool): Whether to rescale the intensity of the image.
        sobel_filter (bool): Apply Sobel filter to detect edges in the image
        wavelets (bool): Apply wavelet transform to the image.
        GLCM (bool): Compute the Gray Level Co-occurrence Matrix (GLCM) for texture analysis.
        GLCM_distance (int): The distance between pixel pairs for GLCM calculation.
        GLCM_levels (int): The number of gray levels to be used in the GLCM computation.
    """
    if image is None:
        image = open_dcm(image_path)

    if gaussian_smoothing:
        image = gaussian_filter(image, sigma = 1, radius = 5)
    
    if wavelets:
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
        image = cA
        cA = np.clip(cA, 0, 255)
        cH = np.clip(cH, 0, 255)
        cV = np.clip(cV, 0, 255)
        cD = np.clip(cD, 0, 255)
        image = [cA, cH, cV, cD]
    
    if extract_window_lesion and image_path is not None and wavelets is False:
        image = extract_window(image_path=image_path, image=image, image_folder='', metadata_file="data_example/Runs_dataset.csv")
    elif extract_window_lesion and image_path is not None and wavelets is True:
        for i in range(len(image)):
            image[i] = extract_window(image_path=image_path, image=image[i], image_folder='', wavelets=True, metadata_file="data_example/Runs_dataset.csv")

    #return the preprocessed image
    if image is not None and len(image) > 0:
        return image
    else:
        raise ValueError("Invalid combination of filters and techniques.")
    
def generate_all_parameters_combinations():
    grid = {
        # Lissage
        "gaussian_smoothing": [True, False],
        # Ondelettes
        "wavelets": [True, False]
    }
    
    base_keys = ["gaussian_smoothing", "wavelets"]
    base_combinations = [dict(zip(base_keys, v)) for v in itertools.product(*[grid[k] for k in base_keys])]
    
    final_combinations = []
    for base_combination in base_combinations:
        temp_comb = base_combination.copy()
        for key in grid.keys():
            if key not in temp_comb:
                temp_comb[key] = None
        
        final_combinations.append(temp_comb)
    
    return final_combinations



if __name__ == "__main__":
    path = "data_example/MRIs/Patient_1/MRI_1/export_00062.DCM"
    original_image = open_dcm(path)
    
    parameters_choice = {'gaussian_smoothing': True, 'wavelets': False}
# or
    # # Preprocess the image with all possible parameters combinations and display results
    # parameters_generation = generate_all_parameters_combinations() 
    
    # for i, comb in enumerate(parameters_generation):
    #     print(f"Combinaison {i+1}: {comb}")
    # # Sélection aléatoire d'une combinaison
    # parameters_choice = random.choice(parameters_generation)
    
    
    # Application du prétraitement
    image = preprocessing_image(image_path=path,extract_window_lesion=True, **parameters_choice)
    
    if isinstance(image, list): 
        if len(image[0].shape) ==4:
            fig, ax = plt.subplots(4, 12, figsize=(10, 5))
            
            for i in range(4):  # 4 lignes
                for j in range(12):  # 12 colonnes
                    k = j % 3  # On boucle sur 0,1,2 pour les canaux
                    l = j // 3  # On boucle sur 0,1,2,3 pour l'autre dimension

                    glcm = image[i][:, :, k, l]  # Extraire la bonne matrice GLCM
                    ax[i, j].imshow(glcm, cmap='gray')
                    ax[i, j].axis('off')
                    ax[i, j].set_title(f'Distance: {k+1}\n Angle: {l*45}°')
                    
                
        else: #list of wavelets
            figu, ax = plt.subplots(1, 5, figsize=(20, 5))
            ax[0].imshow(original_image, cmap='gray')
            ax[0].set_title("Original Image")
            ax[1].imshow(image[0], cmap='gray')
            ax[1].set_title("cA Image")
            ax[2].imshow(image[1], cmap='gray')
            ax[2].set_title("cH Image")
            ax[3].imshow(image[2], cmap='gray')
            ax[3].set_title("cV Image")
            ax[4].imshow(image[3], cmap='gray')
            ax[4].set_title("cD Image")
    elif isinstance(image, np.ndarray):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title("Original Image")
        ax[1].imshow(image, cmap='gray')
        ax[1].set_title("Preprocessed Image")
    else:
        raise ValueError("Invalid output type")
    plt.tight_layout()
    plt.show()
