# Import required packages
import numpy as np
from scipy import ndimage as ndi
import skimage
import os
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from pathlib import Path

def process_directory(input_dir, output_dir, file_ext=[".tif"], thresh=0.99, cell_type="EB"):

    # Validate the inputs
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    elif isinstance(input_dir, Path):
        pass
    else:
        raise ValueError(f"Expected the first argument to be a str or Path to the input directory. Instead it is a {type(input_dir)}.")
    
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise ValueError(f"Expected the second argument to be a str or Path to the output directory. Instead it is a {type(output_dir)}.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    elif output_dir.is_file():
        raise ValueError(f"Expected the second argument to a path to a directory. Instead it appears to be a file.")

    # Get list of files
    all_files = input_dir.rglob("*")
    file_list = []
    for f in all_files:
        if f.suffix in file_ext:
            file_list.append(f.resolve())

    with tqdm(file_list) as pbar:
        for f in pbar:
            pbar.set_description(f"{f.name}")
            process_image(f, output_dir, thresh=thresh, cell_type=cell_type)

def process_image(input_path, output_dir, thresh=0.99, cell_type="EB"):

    # Read in image
    image_rgb = skimage.io.imread(input_path)
    image_gray = skimage.color.rgb2gray(image_rgb)

    match cell_type:
        case "EB":
            labels, inner_cell_labels = segment_cells(image_gray, thresh=thresh)

        case "ES":
            labels, inner_cell_labels = segment_cells_dark(image_gray, thresh=thresh)
    
    # Measure properties
    cell_props = skimage.measure.regionprops(labels)

    if inner_cell_labels:
        inner_cell_props = skimage.measure.regionprops(inner_cell_labels)

    mean_distances = []

    for p in tqdm(cell_props, leave=False):
        
        # Calculate the average thickness of bright regions
        curr_cell_mask = np.zeros(labels.shape, dtype=np.bool)
        curr_cell_mask[labels == p['label']] = True

        if inner_cell_labels:
            contours = skimage.measure.find_contours(curr_cell_mask)

            # Return the longest contour
            longest = sorted(contours, key=len, reverse=True)[:1]
            longest = np.array(longest[0], dtype=int)

            curr_inner_mask = np.zeros(inner_cell_labels.shape, dtype=np.bool)
            curr_inner_mask[inner_cell_labels == p['label']] = True

            # Make a mask that leaves only the center region false
            curr_inner_mask_bg_filled = curr_inner_mask + (labels != p['label'])
            curr_inner_mask_bg_filled = skimage.morphology.remove_small_holes(curr_inner_mask_bg_filled, max_size=500)

            curr_dist = ndi.distance_transform_edt(curr_inner_mask_bg_filled)

            mean_distances.append(np.mean(curr_dist[longest[:, 0], longest[:, 1]]))

    # Save output
    fn = input_path.stem
    with open(os.path.join(output_dir, fn + ".csv"), 'w', newline='') as file:
    
        writer = csv.writer(file, delimiter=",")

        if inner_cell_labels:
    
            #Write CSV headers
            writer.writerow(["Cell", "Label", "Total area (px)", "Bright region area (px)", "Ratio (Bright/Total)", "Mean Thickness (px)"])
        
            ctr = 0
            for p in cell_props:
                writer.writerow([ctr + 1, p.label, p.area, inner_cell_props[ctr].area, inner_cell_props[ctr].area/p.area, mean_distances[ctr]])
                ctr += 1

        else:

            #Write CSV headers
            writer.writerow(["Cell", "Label", "Total area (px)"])
        
            ctr = 0
            for p in cell_props:
                writer.writerow([ctr + 1, p.label, p.area])
                ctr += 1


    if inner_cell_labels:
        ovimg = imoverlay(image_rgb, inner_cell_labels, [0, 1, 0, 0.4], plot_outlines=False)

    else:
        ovimg = imoverlay(image_rgb, labels, [0, 1, 0, 0.4], plot_outlines=False)

    skimage.io.imsave(os.path.join(output_dir, fn + ".png"), ovimg)
    # exit()

def segment_cells(image, thresh=0.99):
    
    mask = image < (thresh * np.max(image))  # Note: Most images have a saturated white background

    mask = skimage.morphology.opening(mask, skimage.morphology.disk(30))
    mask = skimage.morphology.remove_small_holes(mask, max_size=200)
    
    # Watershed
    distance = ndi.distance_transform_edt(mask)
    coords = skimage.feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask, threshold_abs=(0.5 * np.max(distance)), min_distance=75)

    mask_marker = np.zeros(distance.shape, dtype=bool)
    mask_marker[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_marker)
    labels = skimage.segmentation.watershed(-distance, markers, mask=mask)
    
    labels = skimage.segmentation.clear_border(labels)

    # Decide size cutoff
    props = skimage.measure.regionprops_table(labels, properties=("area",))
    mean_area = np.mean(props["area"])
    stdev_area = np.std(props["area"])
    
    min_area = mean_area - (7 * stdev_area)
    # print(min_area)
    # print(mean_area)
    
    labels = skimage.morphology.remove_small_objects(labels, max_size=min_area)

    # output_test = skimage.segmentation.mark_boundaries(image, labels, mode="thick", color=(0, 1, 0))

    # plt.imshow(output_test)
    # plt.show()
    # exit()

    # Do a global threshold to determine dark/light threshold
    thresh_cell = skimage.filters.threshold_otsu(image[labels > 0])
   
    inner_cell_mask = image >= thresh_cell
    
    inner_cell_labels = labels.copy()
    inner_cell_labels[~inner_cell_mask] = 0

    return (labels, inner_cell_labels)

def segment_cells_dark(image, thresh=0.99):

    mask = image > (thresh * np.max(image))

    # mask = skimage.morphology.opening(mask, skimage.morphology.disk(30))
    mask = skimage.morphology.remove_small_holes(mask, max_size=100000)
    mask = skimage.morphology.opening(mask, skimage.morphology.disk(5))
   
    # Watershed
    distance = ndi.distance_transform_edt(mask)
    coords = skimage.feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask, threshold_abs=(0.3 * np.max(distance)), min_distance=75)

    mask_marker = np.zeros(distance.shape, dtype=bool)
    mask_marker[tuple(coords.T)] = True
    
    # # To debug watershed
    # output_test = imoverlay(image, mask, color=[0, 1, 0, 0.4], plot_outlines=False)

    # plt.imshow(output_test)
    # plt.plot(coords[:, 1], coords[:, 0], 'rx')
    # plt.show()
    # exit()

    markers, _ = ndi.label(mask_marker)
    labels = skimage.segmentation.watershed(-distance, markers, mask=mask)
    labels = skimage.segmentation.clear_border(labels)

    # Decide size cutoff
    props = skimage.measure.regionprops_table(labels, properties=("area",))
    mean_area = np.mean(props["area"])
    stdev_area = np.std(props["area"])
    
    min_area = mean_area - (3 * stdev_area)
    # print(min_area)
    # print(mean_area)
    
    labels = skimage.morphology.remove_small_objects(labels, max_size=min_area)

    output_test = imoverlay(image, labels, color=[0, 1, 0, 0.4], plot_outlines=False)

    # plt.imshow(output_test)
    # plt.show()
    # exit()

    # # Do a global threshold to determine dark/light threshold
    # thresh_cell = skimage.filters.threshold_otsu(image[labels > 0])
   
    # inner_cell_mask = image >= thresh_cell
    
    # inner_cell_labels = labels.copy()
    # inner_cell_labels[~inner_cell_mask] = 0

    return (labels, None)

# Define function to generate overlay images
def imoverlay(image_A, image_B, color, plot_outlines=True, normalize=True):
    # Always assume that image_A is supposed to be an image
    # Image_B can be an image, binary mask, or label

    # if normalize:
    #     if image_A.ndims == 1:
    #         image_A = 
    #     for c in range(image_A)
    

    if plot_outlines and (image_B.ndim == 2):
        image_B = skimage.segmentation.find_boundaries(image_B)
    else:
        image_B = image_B > 0
    # plt.imshow(outlines)

    image_out = np.zeros((image_A.shape[0], image_A.shape[1], 3), np.uint8)

    for c in range(3):
        if image_A.ndim < 3:
            curr_slice = (image_A - np.min(image_A))/(np.max(image_A) - np.min(image_A)) * 255
        else:
            curr_slice = image_A[:, :, c]            

        if len(color) < 4:
            alpha = 1
        else:
            alpha = color[3]
            
        curr_slice[image_B] = (color[c] * 255 * alpha) + ((1 - alpha) * curr_slice[image_B])
        image_out[:, :, c] = curr_slice

    return image_out

if __name__ == "__main__":

    # data_directory = r"\\pn.vai.org\projects_secondary\wen\vari-core-generated-data\OIC\Junwei 04302026\EB organoid"

    # output_directory = "../processed/2026-05-12 EB organoid"
    
    # process_directory(data_directory, output_directory)

    data_directory = r"\\pn.vai.org\projects_secondary\wen\vari-core-generated-data\OIC\Junwei 04302026\ES cell"

    output_directory = "../processed/2026-05-12 ES cell"
    
    process_directory(data_directory, output_directory, thresh=0.65, cell_type="ES")


    data_directory = r"../data/ES cell/Set 2"

    output_directory = "../processed/2026-05-12 ES cell"
    
    process_directory(data_directory, output_directory, thresh=0.45, cell_type="ES")

    # image_path = r"D:\Projects\OIC-303 Junwei\data\ES cell\WT19-1 10X.tif"

    # output_directory = "../processed/2026-05-12 ES cell_DEV"
    
    # process_image(image_path, output_directory, cell_type="ES")