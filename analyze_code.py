# Import required packages
import numpy as np
from scipy import ndimage as ndi
import skimage
import os
import matplotlib.pyplot as plt
import csv
from core_utils import imoverlay
import segmentation
from tqdm import tqdm

# Specify input/output directories
# data_directory = '\\\\pn.vai.org\\projects\\wen\\vari-core-generated-data\\OIC\\OIC-234 EB Junwei\\EB8 image'
# output_directory = "\\\\pn.vai.org\\projects\\wen\\vari-core-generated-data\\OIC\\OIC-234 EB Junwei\\EB8 image\\Measurements"

#data_directory = '../data/EB8 image'
#output_directory = '../processed/20251228'

data_directory = '\\\\pn.vai.org\\projects\\wen\\vari-core-generated-data\\OIC\\OIC 01272026 Junwei'

output_directory = '\\\\pn.vai.org\\projects\\wen\\vari-core-generated-data\\OIC\\OIC 01272026 Junwei\\Measurements'

# wt13-1
# wt13-2
# wt13-3
# wt13-4
# wt14-2
# wt14-3
# wt14-4
# wt14-5
# wt23-2

# Begin processing

os.makedirs(output_directory, exist_ok=True)

files_list = os.listdir(data_directory)

for f in tqdm(files_list):

    file = os.path.join(data_directory, f)
    
    if os.path.isfile(file):
        #print(f"Processing file: {file}")

        # Determine output filename
        fn = os.path.splitext(os.path.basename(file))[0]

        # Read in image
        image = skimage.io.imread(file)
        image = skimage.color.rgb2gray(image)
        
        # Identify the EB cells

        if (fn == "wt13-1") or (fn == "HET66-2 2X") or (fn == "HET66-3 2X"):
            # Handle these images
            labels, inner_cell_labels = segmentation.segment_cells(image, thresh = 0.85)
        else:        
            continue # Skip remaining files for now
            # labels, inner_cell_labels = segmentation.segment_cells(image)

        # Measure properties
        cell_props = skimage.measure.regionprops(labels)
        inner_cell_props = skimage.measure.regionprops(inner_cell_labels)

        mean_distances = []

        for p in tqdm(cell_props, leave=False):
             
            # Calculate the average thickness of bright regions
            curr_cell_mask = np.zeros(labels.shape, dtype=np.bool)
            curr_cell_mask[labels == p['label']] = True

            contours = skimage.measure.find_contours(curr_cell_mask)

            # Return the longest contour
            longest = sorted(contours, key=len, reverse=True)[:1]
            longest = np.array(longest[0], dtype=int)

            curr_inner_mask = np.zeros(inner_cell_labels.shape, dtype=np.bool)
            curr_inner_mask[inner_cell_labels == p['label']] = True

            # Make a mask that leaves only the center region false
            curr_inner_mask_bg_filled = curr_inner_mask + (labels != p['label'])
            curr_inner_mask_bg_filled = skimage.morphology.remove_small_holes(curr_inner_mask_bg_filled, 500)

            curr_dist = ndi.distance_transform_edt(curr_inner_mask_bg_filled)

            mean_distances.append(np.mean(curr_dist[longest[:, 0], longest[:, 1]]))


        # Save output
        with open(os.path.join(output_directory, fn + ".csv"), 'w', newline='') as file:
        
            writer = csv.writer(file, delimiter=",")   
        
            #Write CSV headers
            writer.writerow(["Cell", "Label", "Total area (px)", "Bright region area (px)", "Ratio (Bright/Total)", "Mean Thickness (px)"])
        
            ctr = 0
            for p in cell_props:
                writer.writerow([ctr + 1, p.label, p.area, inner_cell_props[ctr].area, inner_cell_props[ctr].area/p.area, mean_distances[ctr]])
                ctr += 1
                
        ovimg = imoverlay(image, inner_cell_labels, [0, 1, 0, 0.4], plot_outlines=False)
        skimage.io.imsave(os.path.join(output_directory, fn + ".png"), ovimg)

        #print('\b...DONE', flush=True)