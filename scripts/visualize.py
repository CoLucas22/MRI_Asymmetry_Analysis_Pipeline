from utils import divide_image,open_dcm
from extract_features import calculate_difference_old, find_best_window, summarise_rows, find_best_2_rows, smoothing_scores
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.ndimage import rotate

def display_halves_and_scores(image):
    right_half, left_half = divide_image(image)

    JSD, MAE, DSSIM = calculate_difference_old(left_half, right_half)

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



def display_slice_and_scores_profile(image_path, scores_df_path = "data_example/data_aggregated.csv"):
    scores_df = pd.read_csv(scores_df_path, index_col=0)
    quantiles_df, symmetry_axis_points, window_size, angle, index_debut, index_fin = find_best_window(image_path=image_path, df=scores_df)

    smoothed_df = smoothing_scores(quantiles_df, image_path)

    quantiles_df = smoothed_df
    quantiles_df = pd.concat([quantiles_df, quantiles_df], ignore_index=False).sort_index()
    
    quantiles_array, _ = summarise_rows(quantiles_df, method="meanof3bests")  
    best_row, second_row = find_best_2_rows(quantiles_array)

    mpl.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    fig, ax = plt.subplots(
        1, 2,
        figsize=(10, 8),
        gridspec_kw={'width_ratios': [2, 1]}  # Image = 2x plus large que le plot
    )
    
    image = open_dcm(image_path)
    image = rotate(image, angle, reshape=False)
        # --- IMAGE ---
    ax[0].imshow(image, cmap='gray')
    ax[0].axhline(y=best_row, color='#ffcc00', linestyle='--', linewidth=2,
                label=f"Best row {best_row}; mean: {round(quantiles_array[best_row], 2)}")
    ax[0].axhline(y=second_row, color='#00cc66', linestyle='--', linewidth=2,
                label=f"Second best row {second_row}; mean: {round(quantiles_array[second_row], 2)}")
    ax[0].axvline(x=symmetry_axis_points[0][0], color='red', linestyle='--', linewidth=1,
                label=f"Symmetry axis at x={symmetry_axis_points[0][0]}")
    window_center = symmetry_axis_points[0][0]
    window_height = int(window_size[0])
    window_width = int(window_size[1])
    ax[0].add_patch(plt.Rectangle(
        (window_center - window_height // 2, best_row - window_width // 2),
        window_height, window_width,
        edgecolor='#3399ff', facecolor='none', lw=2, linestyle='-',
        label="Best patch"
    ))

    ax[0].set_title("DICOM image")
    ax[0].axis("off")
    ax[0].legend(loc='lower right', frameon=True)

    # --- COURBES ---
    colors = plt.cm.tab10.colors
    for i, col in enumerate(quantiles_df.columns):
        if col != "position":
            ax[1].plot(quantiles_df[col], range(len(quantiles_df)),
                    label=col, alpha=0.6, color=colors[i % len(colors)], linewidth=1.5)

    ax[1].axhline(y=best_row, color='#ffcc00', linestyle='--', linewidth=2)
    ax[1].axhline(y=second_row, color='#00cc66', linestyle='--', linewidth=2)

    ax[1].invert_yaxis()
    #ax[1].set_xlabel("Valeurs")
    #ax[1].set_ylabel("Position (ligne)")
    ax[1].set_title("Mean of quantiles")
    ax[1].grid(True, linestyle=':', alpha=0.4)
    ax[1].legend(frameon=True)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    image_path = "data_example/MRIs/Patient_1/MRI_1/export_00062.DCM"
    image = open_dcm(image_path)
    #display_halves_and_scores(image)
    
    display_slice_and_scores_profile(image_path, scores_df_path="data_example/data_aggregated.csv")