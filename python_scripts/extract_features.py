import numpy as np
from scipy.spatial.distance import jensenshannon
from utils import open_dcm, divide_image
import pandas as pd
from utils import sliding_window_on_axis
from preprocess import preprocessing_image, window_size_in_pixels, symmetry_by_k
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
from scipy.stats import beta


def ssim_formula(img1, img2, k1=0.01, k2=0.03, L=255):
    """
    Calcule l'indice SSIM entre deux images
    """
    if np.std(img1) < 1e-10 or np.std(img2) < 1e-10:
        return 0.0  # Retourner une valeur par défaut si l'une des images est constante
    
    # Paramètres du SSIM
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    
    # Moyennes
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Variances et covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    
    # Utiliser try/except pour gérer les erreurs de cov
    try:
        # Ajouter un epsilon à img1 et img2 si nécessaire pour éviter les divisions par zéro
        epsilon = np.finfo(float).eps
        
        # Si les écarts-types sont trop proches de zéro, ajouter du bruit
        if sigma1_sq < epsilon:
            img1 = img1 + np.random.normal(0, epsilon, img1.shape)
            sigma1_sq = np.var(img1)
        
        if sigma2_sq < epsilon:
            img2 = img2 + np.random.normal(0, epsilon, img2.shape)
            sigma2_sq = np.var(img2)
        
        # Calculer la covariance avec les images potentiellement modifiées
        sigma12 = np.cov(img1.ravel(), img2.ravel())[0, 1]
    except:
        # En cas d'échec, utiliser une approche alternative
        sigma12 = np.sum((img1 - mu1) * (img2 - mu2)) / (img1.size - 1)
    
    # Formule SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    # Éviter la division par zéro
    if denominator < epsilon:
        return 0.0
        
    ssim = numerator / denominator
    return ssim

def calculate_difference_old(left_half, right_half, JS_hist_bins="Sturges", ssim=True):
    
    # Convertir les NaN en 0 et spécifier le type float64
    left_half = np.fliplr(np.nan_to_num(left_half, nan=0.0)).astype(np.float64)
    right_half = np.nan_to_num(right_half, nan=0.0).astype(np.float64)

    # Vérifier si les matrices sont vides
    if left_half.size == 0 or right_half.size == 0:
        return 0,0,0,0

    # Adapter les dimensions si nécessaire
    if left_half.shape != right_half.shape:
        smallest_shape = tuple(min(dim1, dim2) for dim1, dim2 in zip(left_half.shape, right_half.shape))

        # Slice each array to match the smallest shape
        left_half = left_half[tuple(slice(0, s) for s in smallest_shape)]
        right_half = right_half[tuple(slice(0, s) for s in smallest_shape)]

    # Calculate Jensen-Shannon divergence
    nb_bins = int(np.log2(left_half.size)) + 1 if JS_hist_bins == "Sturges" else int(left_half.size ** 0.25) if JS_hist_bins == "Yule" else int(JS_hist_bins)
    hist_range = (0, 255)
    
    # Vérifier s'il y a des données valides pour l'histogramme
    if np.all(left_half == 0) or np.all(right_half == 0):
        #print("Les deux moitiés sont vides.")
        JSD = 0
    else:
        hist_left, _ = np.histogram(left_half.ravel(), bins=nb_bins, range=hist_range, density=False)
        hist_right, _ = np.histogram(right_half.ravel(), bins=nb_bins, range=hist_range, density=False)
        
        # Éviter les divisions par zéro dans jensenshannon
        # En ajoutant une petite valeur aux histogrammes vides
        epsilon = np.finfo(float).eps
        hist_left = hist_left + epsilon
        hist_right = hist_right + epsilon
        
        # JS distance
        JSD = jensenshannon(hist_left, hist_right)** 2

    # SSIM calculation
    if ssim:
        try:
            DSSIM = (1 - ssim_formula(img1=left_half, img2=right_half)) / 2
        except:
            DSSIM = 0
    else:
        DSSIM = 0

    #Mean Absolute Error
    max_val_left = np.max(left_half)
    max_val_right = np.max(right_half)
    
    if max_val_left > 0:
        left_half_norm = left_half / max_val_left
    else:
        left_half_norm = left_half
        
    if max_val_right > 0:
        right_half_norm = right_half / max_val_right
    else:
        right_half_norm = right_half

    diff = left_half_norm - right_half_norm
    
    # Éviter la division par zéro
    if left_half.size > 0:
        MAE = np.sum(np.abs(diff)) / left_half.size
    else:
        MAE = np.nan

    return JSD, MAE, DSSIM

def display_halves_and_scores(image):
    right_half, left_half = divide_image(image)

    JSD, MAE, DSSIM = calculate_difference_old(left_half, right_half)

    #plot both halves and scores
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(left_half, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title(f"Left Half\nJSD: {JSD:.4f}\nMAE: {MAE:.4f}\nDSSIM: {DSSIM:.4f}")
    axs[1].imshow(right_half, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title(f"Right Half\nJSD: {JSD:.4f}\nMAE: {MAE:.4f}\nDSSIM: {DSSIM:.4f}")
    # Bar plot with value labels and colors
    scores = [JSD, MAE, DSSIM]
    labels = ['JSD', 'MAE', 'DSSIM']
    colors = ['#4C72B0', '#55A868', '#C44E52']
    bars = axs[2].bar(labels, scores, color=colors, edgecolor='black')
    axs[2].set_ylim(0, max(scores) * 1.2 if max(scores) > 0 else 1)
    axs[2].set_title("Scores", fontsize=14, fontweight='bold')
    axs[2].set_ylabel("Value")
    axs[2].grid(axis='y', linestyle='--', alpha=0.6)
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        axs[2].annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.show()

def beta_weights(x, a=5, b=2, start=0.2, end=0.8):
    """
    Calcule des poids basés sur une loi beta avec des bornes ajustables et un retour exact à 0 à l'index spécifié.
    
    Parameters:
    -----------
    x : array-like
        Grille de points sur laquelle calculer les poids
    a : float
        Premier paramètre de la loi beta
    b : float
        Second paramètre de la loi beta
    start : float
        Borne inférieure pour la montée des poids (0 <= start < end <= 1)
    end : float
        Borne supérieure pour la descente des poids (0 <= start < end <= 1)
    
    Returns:
    --------
    weights : numpy.ndarray
        Poids calculés sur la grille x
    """

    x = np.linspace(0, len(x), len(x))  
    # Initialisation des poids à zéro
    weights = np.zeros_like(x, dtype=float)
    
    # On applique la loi beta uniquement dans l'intervalle [start, end]
    mask = (x >= start) & (x <= end)
    
    # Calculer les poids beta
    beta_weights = beta.pdf((x[mask] - start) / (end - start), a, b)
    
    # Appliquer une fenêtre pour un retour exact à 0 à l'index spécifié
    window = (1 - ((x[mask] - start) / (end - start)) ** 2) * (x[mask] <= end)
    weights[mask] = beta_weights * window
    
    # Normalisation
    if np.max(weights) > 0:
        weights /= np.max(weights)
    return weights

def vertical_weighting(MAE, DSSIM, JSD, image_path):
    """
    Calcule des poids selon une courbe exponentielle qui commence après 1 cm de distance.
    
    Arguments:
        MAE, DSSIM, JSD: Trois listes de métriques
        image_path: Chemin vers l'image pour obtenir l'espacement
    
    Returns:
        Une liste de scores de pondération
    """
    # Lecture de l'image et de son espacement
    image = sitk.ReadImage(image_path)
    spacing = image.GetSpacing()
    # Si l'image est 2D, prendre le premier espacement
    y_spacing = spacing[1] if len(spacing) > 2 else spacing[0]  # en mm
    # Conversion de 1 cm en nombre de tranches
    tranches_par_cm = int(10 // y_spacing)  # 10mm = 1cm
    longueur_liste = len(MAE)
    scores = [0] * longueur_liste
    # Fonction pour vérifier si les 3 scores sont > 0 sur 3 lignes consécutives
    def triple_positive(i):
        return all(
            all(x > 0 for x in (MAE[j], DSSIM[j], JSD[j]))
            for j in range(i, i + 3)
        )
    # Trouver l'index de début : première position où les 3 lignes consécutives sont valides
    index_debut_ascension = next(
        (i for i in range(longueur_liste - 2) if triple_positive(i)),
        None
    )
    # Trouver l'index de fin : dernière position où les 3 lignes consécutives sont valides
    index_fin_descente = next(
        (i for i in range(longueur_liste - 3, -1, -1) if triple_positive(i)),
        None
    )
    # Gestion des cas limites
    if index_debut_ascension is None or index_fin_descente is None or index_debut_ascension >= index_fin_descente:
        return scores

    index_debut_croissance = min(index_debut_ascension + tranches_par_cm, index_fin_descente - 1)
    
    # # S'assurer que le début de croissance ne dépasse pas la fin de descente
    index_debut_croissance = min(index_debut_croissance, index_fin_descente - 1)
    scores = beta_weights(MAE, a=5, b=2, start=index_debut_croissance, end=index_fin_descente)
    
    return scores, index_debut_croissance*2, index_fin_descente*2

def extract_corresponding_ref_quantile(data, column_name, test_score=0, plot= False):
    """Extract the exact quantile of test_score in the distribution of column_name for label 0
    and plot the distribution with a vertical line at test_score.
    """

    # Extraire les valeurs de column_name pour label 0
    values = data[data['label'] == 0][column_name].values

    # Calculer le quantile exact
    quantile = (values < test_score).mean() * 100 # Proportion des valeurs inférieures à test_score

    if plot:
    # Plot de la distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(values, bins=50, kde=True, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(test_score, color='red', linestyle='dashed', linewidth=2, label=f'Score: {test_score}')
        plt.xlabel(column_name)
        plt.ylabel('Fréquence')
        plt.title(f'Distribution de {column_name} (label=0) avec test_score')
        plt.legend()    
        plt.show()

    return quantile

def compute_all_windows_scores_quantiles(image, axis_points:list, window_size:tuple, data, ssim = False, image_used = "cH"):

    #prendre un point sur 2 de l'axe
    axis_points = axis_points[::2]

    if image_used == "cH" or image_used == "cV" or image_used == "cD":
        #diviser par 2 la taille de fenetre
        window_size = (window_size[0]//2, window_size[1]//2)
    ref_JS = np.zeros(len(axis_points)) 
    ref_MAE = np.zeros(len(axis_points))
    ref_DSSIM = np.zeros(len(axis_points))
    #initiate scores to 0
    liste_J_S = np.zeros(len(axis_points))
    liste_sumdiff = np.zeros(len(axis_points))
    liste_dssim = np.zeros(len(axis_points))
    liste_positions = np.zeros((len(axis_points), 2))
    
    if image_used == "cH" or image_used == "cV" or image_used == "cD":
        axis_points = [(x, y//2) for x, y in axis_points]

    axis_set = {y for _, y in axis_points}
        
    
    for idx, (x, y, window) in enumerate(sliding_window_on_axis(image, axis_points, window_size)):
            
        if y in axis_set:
            right_half, left_half = divide_image(window)
              #Calculate the difference between the two halves
            scoreJS, sum_diff, DSSIM = calculate_difference_old(left_half, right_half, JS_hist_bins=15, ssim=ssim)

            if image_used == "cH":
                jsd = extract_corresponding_ref_quantile(data, column_name='cH_JSD', test_score=scoreJS)
                liste_J_S[idx] = jsd
                dssim = extract_corresponding_ref_quantile(data, column_name='cH_DSSIM', test_score=DSSIM)
                liste_dssim[idx] = dssim
            elif image_used == "cV"  :

                jsd = extract_corresponding_ref_quantile(data, column_name='cH_JSD', test_score=scoreJS)
                liste_J_S[idx] = jsd
                dssim = extract_corresponding_ref_quantile(data, column_name='cV_DSSIM', test_score=DSSIM)
                liste_dssim[idx] = dssim

            elif image_used == "base":
                ref_DSSIM[idx] = DSSIM
                ref_JS[idx] = scoreJS
                ref_MAE[idx] = sum_diff
                liste_dssim[idx] = extract_corresponding_ref_quantile(data, column_name='DSSIM', test_score=DSSIM)
                liste_J_S[idx] = extract_corresponding_ref_quantile(data, column_name='JSD', test_score=scoreJS)
                liste_sumdiff[idx] = extract_corresponding_ref_quantile(data, column_name='MAE', test_score=sum_diff)

            liste_positions[idx] = (x, y)
        else:
            liste_positions[idx] = (0, 0)
    plt.close('all')  # Ferme toutes les fenêtres ouvertes
    if image_used == "base":
        # Compute the final scores
        return liste_J_S, liste_sumdiff, liste_dssim, ref_DSSIM, ref_JS, ref_MAE, liste_positions

    else:
        return liste_J_S, liste_sumdiff, liste_dssim, liste_positions

def find_best_window(image_path, df):
    (mean_angle, x_axis), image, angle = symmetry_by_k(image_path, k=8)
    symmetry_axis_points = [(x_axis, i) for i in range(image.shape[0])]
    
    quantiles_windows_df = pd.DataFrame(columns = ['cH_DSSIM',"cV_DSSIM","cH_JSD","cV_JSD","JSD","MAE", "DSSIM", "position"])

    window_size = window_size_in_pixels(image_path, window_height_cm=1, window_width_cm=7)
    del image
    
    images_to_process = ["base", "cH", "cV"]
    
    for key in images_to_process:
        if key == "cH":
            image = preprocessing_image(image_path, gaussian_smoothing=True, wavelets=True)[0]
            quantiles_list_temporary_JSD, _, quantiles_list_temporary_DSSIM, _ = compute_all_windows_scores_quantiles(image, symmetry_axis_points, window_size, df, ssim= True, image_used= key)
            quantiles_windows_df["cH_JSD"] = list(map(int,quantiles_list_temporary_JSD))
            quantiles_windows_df["cH_DSSIM"] = list(map(int,quantiles_list_temporary_DSSIM))
            
        elif key == "cV":
            image = preprocessing_image(image_path, gaussian_smoothing=False, wavelets=True)[1]
            quantiles_list_temporary_JSD, _, quantiles_list_temporary_DSSIM, _ = compute_all_windows_scores_quantiles(image, symmetry_axis_points, window_size, df, ssim= False, image_used= key)
            quantiles_windows_df["cV_JSD"] = quantiles_list_temporary_JSD
            quantiles_windows_df["cV_DSSIM"] = quantiles_list_temporary_DSSIM


        elif key == "base":
            image = preprocessing_image(image_path, gaussian_smoothing=False, wavelets=False)
            quantiles_list_temporary_JSD, quantiles_list_temporary_MAE, quantiles_list_temporary_DSSIM, ref_DSSIM, ref_JS, ref_MAE, liste_positions= compute_all_windows_scores_quantiles(image, symmetry_axis_points, window_size, df, ssim= True, image_used= key)
            quantiles_windows_df["DSSIM"] = quantiles_list_temporary_DSSIM
            quantiles_windows_df["JSD"] = quantiles_list_temporary_JSD
            quantiles_windows_df["MAE"] = quantiles_list_temporary_MAE
            quantiles_windows_df["position"] = liste_positions.tolist()
        else:
            raise ValueError("Unknown image type")
        
        
    vertical_weights, index_debut, index_fin = vertical_weighting(ref_MAE, ref_DSSIM, ref_JS, image_path=image_path)
    for col in quantiles_windows_df.columns:
        if col != "position":
            quantiles_windows_df[col] = quantiles_windows_df[col] * vertical_weights
    
    
    for col in quantiles_windows_df.columns:
        if col != "position":
            quantiles_windows_df[col] = quantiles_windows_df[col].map(int)
    
    return quantiles_windows_df, symmetry_axis_points, window_size, angle, index_debut, index_fin

def smoothing_scores(df, image_path):
    """
    Applique un lissage sur chaque colonne numérique d'un DataFrame pandas, en utilisant une fenêtre de 2 cm.
    Pour chaque valeur dans une colonne, calcule la moyenne des voisins dans une plage de 2 cm.
    Gère les cas limites en réduisant la taille de la fenêtre près des bords.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        image_path (str): Chemin vers l'image SimpleITK pour obtenir l'espacement des voxels.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes numériques lissées.
    """

    image = sitk.ReadImage(image_path)
    spacing = np.array(image.GetSpacing())  # Espacement des voxels (x, y, z)
    
    # On suppose que l'on lisse suivant la première dimension de l'image (l'axe des lignes du dataframe)
    spacing_used = spacing[0]
    number_of_lines = df.shape[0]
    #l'espacement de 2 lignes est de 2*spacing_used
    number_of_lines_in_2cm = int(2.0 / spacing_used)*10  # 2 cm / espacement des voxels
    
    smoothed_df = df.copy()

    for col in smoothed_df.select_dtypes(include=[np.number]).columns:
        smoothed_col = smoothed_df[col].copy()
        for i in range(len(smoothed_col)):
            # Calcul des indices de début et de fin de la fenêtre
            start = max(0, i - number_of_lines_in_2cm // 2)
            end = min(len(smoothed_col), i + number_of_lines_in_2cm // 2 + 1)

            smoothed_col.iloc[i] = np.mean(smoothed_col.iloc[start:end])
        smoothed_df[col] = smoothed_col

    return smoothed_df

def summarise_rows(quantiles_df, method="meanof3bests"):
    """
    Summarise the quantiles for each row in the DataFrame.

    Parameters:
    quantiles_df (pd.DataFrame): DataFrame where each row represents an individual 
                                 and each column a quantile.

    Returns:
    a numpy array: Array of summarised quantiles for each row.
    """
    positions_df = quantiles_df["position"]
    quantiles_df = quantiles_df.drop(columns=["position"])
    if method == "mean":
        return quantiles_df.mean(axis=1).values
    elif method == "median":
        return quantiles_df.median(axis=1).values
    elif method == "mean_without_min_max":
        return quantiles_df.apply(lambda row: row.drop([row.idxmin(), row.idxmax()]).mean(), axis=1).values
    elif method == "product":
        return quantiles_df.prod(axis=1).values
    elif method == "thresholds":
        #compute mean of indexes on axis 0 and return (number of indexes > 0.5) / len(indexes)
        means = quantiles_df.mean(axis=0).values
        return quantiles_df.apply(lambda row: (np.sum(row > 0.5) / len(row)), axis=1).values
    elif method == "meanof3bests":
        return quantiles_df.apply(lambda row: row.nlargest(3).mean(), axis=1).values, positions_df
    else:
        raise ValueError("NON")

def find_best_2_rows(quantiles_array):
    """
    Find the two rows with the highest values, ensuring they are 
    separated by at least 10% of the array length.

    Parameters:
    quantiles_array (np.ndarray): 1D NumPy array containing quantile values.

    Returns:
    tuple: Indices of the two selected values.
    """
    #print(quantiles_array.shape)
    
    # Make sure we have enough data points
    if len(quantiles_array) < 2:
        raise ValueError("The array must contain at least two elements.")
    
    # Sort indices by their values in descending order
    sorted_indices = np.argsort(quantiles_array)[::-1]
    
    # Minimum distance required between selected indices
    min_distance = max(1, int(0.1 * len(quantiles_array)))
    
    # Select the best first index
    best_first = sorted_indices[0]
    
    # Find the second index that respects the distance constraint
    for idx in sorted_indices[1:]:
        if abs(best_first - idx) > min_distance:
            return best_first, idx
    
    # If no pair respecting the distance was found, take the second best without constraint
    return best_first, sorted_indices[1]

if __name__ == "__main__":
    path = "data_example/MRIs/Patient_1/MRI_1/export_00062.DCM"
    original_image = open_dcm(path)
    display_halves_and_scores(original_image)
