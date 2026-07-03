# Import required packages
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt

# from cellpose import models
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


def process_directory(
    input_dir,
    output_dir,
    file_ext=[".tif"],
    threshold=0.99,
    cell_type="EB",
    spacing=None,
):

    # Validate the inputs
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    elif isinstance(input_dir, Path):
        pass
    else:
        raise ValueError(
            f"Expected the first argument to be a str or Path to the input directory. Instead it is a {type(input_dir)}."
        )

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise ValueError(
            f"Expected the second argument to be a str or Path to the output directory. Instead it is a {type(output_dir)}."
        )

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    elif output_dir.is_file():
        raise ValueError(
            "Expected the second argument to a path to a directory. Instead it appears to be a file."
        )

    # Get list of files that match the extension(s)
    all_files = input_dir.rglob("*")
    file_list = []
    for f in all_files:
        if f.suffix in file_ext:
            file_list.append(f.resolve())

    all_df = []

    with tqdm(file_list) as pbar:
        for f in pbar:
            pbar.set_description(f"{f.name}")
            df = process_image(
                f,
                output_dir,
                threshold=threshold,
                cell_type=cell_type,
                spacing=spacing,
                pbar=pbar,
                return_df=True,
            )

            # Add the filename
            df["Image"] = str(f)
            all_df.append(df)

    # Merge the dataframes and export
    merged_df = pd.concat(all_df, ignore_index=True)

    export_to_csv(output_dir / "merged.csv", merged_df, spacing=spacing)


def process_image(
    input_path,
    output_dir,
    threshold=0.99,
    cell_type="EB",
    segment_inner=False,
    spacing=None,
    pbar=None,
    return_df=False,
):

    # Validate the inputs
    if isinstance(input_path, str):
        input_path = Path(input_path)
    elif isinstance(input_path, Path):
        pass
    else:
        raise ValueError(
            f"Expected the first argument to be a str or Path to the image file. Instead it is a {type(input_path)}."
        )

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise ValueError(
            f"Expected the second argument to be a str or Path to the output directory. Instead it is a {type(output_dir)}."
        )

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    elif output_dir.is_file():
        raise ValueError(
            "Expected the second argument to a path to a directory. Instead it appears to be a file."
        )

    # Read in image
    image_rgb = skimage.io.imread(input_path)

    # Segment the cell
    update_status(f"{input_path.name}:Segmenting organoids", pbar)
    match cell_type:
        case "EB":
            if segment_inner:
                labels, inner_cell_labels = segment_cells(
                    image_rgb, threshold=threshold, segment_inner=True
                )
            else:
                labels = segment_cells(image_rgb, threshold=threshold)

        case "ES":
            labels, inner_cell_labels = segment_cells_dark(
                image_rgb, threshold=threshold
            )

    # Check if labels are empty
    if labels.max() == 0:
        raise ValueError("Segmentation failed: No object labels were detected.")

    # Measure properties
    update_status(f"{input_path.name}:Measuring properties", pbar)
    props = skimage.measure.regionprops_table(
        labels,
        properties=(
            "label",
            "area",
            "feret_diameter_max",
            "centroid",
            "bbox",
            "image_convex",
        ),
    )

    # Measure internal properties
    # TODO: This is likely broken
    if segment_inner:
        inner_cell_props = skimage.measure.regionprops(inner_cell_labels)

        mean_distances = []

        for p in tqdm(props, leave=False):
            # Calculate the average thickness of bright regions
            curr_cell_mask = np.zeros(labels.shape, dtype=np.bool)
            curr_cell_mask[labels == p["label"]] = True

            if inner_cell_labels:
                contours = skimage.measure.find_contours(curr_cell_mask)

                # Return the longest contour
                longest = sorted(contours, key=len, reverse=True)[:1]
                longest = np.array(longest[0], dtype=int)

                curr_inner_mask = np.zeros(inner_cell_labels.shape, dtype=np.bool)
                curr_inner_mask[inner_cell_labels == p["label"]] = True

                # Make a mask that leaves only the center region false
                curr_inner_mask_bg_filled = curr_inner_mask + (labels != p["label"])
                curr_inner_mask_bg_filled = skimage.morphology.remove_small_holes(
                    curr_inner_mask_bg_filled, max_size=500
                )

                curr_dist = ndi.distance_transform_edt(curr_inner_mask_bg_filled)

                mean_distances.append(np.mean(curr_dist[longest[:, 0], longest[:, 1]]))

    # Save output
    fn = input_path.stem

    # Convert data to DataFrame
    df = pd.DataFrame(props)

    # Drop the centroid information
    df = df.drop(
        columns=[
            "centroid-0",
            "centroid-1",
            "bbox-0",
            "bbox-1",
            "bbox-2",
            "bbox-3",
            "image_convex",
        ]
    )

    export_to_csv(output_dir / (fn + ".csv"), df, spacing=spacing)

    fig = make_labeled_image(image_rgb, labels, props)

    fig.savefig(output_dir / (fn + "_labels.png"))

    update_status(f"{input_path.name}:Data written to {output_dir}.", pbar)

    if return_df:
        return df


def update_status(msg, pbar=None):

    if pbar is not None:
        tqdm.write(msg)
    else:
        print(msg)


# def make_labeled_image(image, labels, props):

#     fig, ax = plt.subplots(figsize=(10, 10))

#     overlay = skimage.segmentation.mark_boundaries(
#         image, labels, mode="thick", color=(1, 0, 1)
#     )

#     ax.imshow(overlay)

#     for idx in range(len(props["label"])):
#         ax.text(
#             props["centroid-1"][idx],
#             props["centroid-0"][idx],
#             props["label"][idx],
#             fontsize=8,
#             color="yellow",
#             fontweight="bold",
#             ha="center",
#             va="center",
#         )

#     return fig


def make_labeled_image(image, labels, props):

    fig, ax = plt.subplots(figsize=(10, 10))

    overlay = skimage.segmentation.mark_boundaries(
        image, labels, mode="thick", color=(1, 0, 1)
    )

    ax.imshow(overlay)

    for idx in range(len(props["label"])):
        ax.text(
            props["centroid-1"][idx],
            props["centroid-0"][idx],
            props["label"][idx],
            fontsize=8,
            color="yellow",
            fontweight="bold",
            ha="center",
            va="center",
        )

        # ---Generated by Gemini---
        # Plot the max feret diameter
        padded_hull = np.pad(
            props["image_convex"][idx], 2, mode="constant", constant_values=0
        )

        contours = skimage.measure.find_contours(
            padded_hull, 0.5, fully_connected="high"
        )
        if not contours:
            continue
        coords = np.vstack(contours)

        distance_matrix = squareform(pdist(coords, "euclidean"))
        idx1, idx2 = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

        pt1_local = coords[idx1]
        pt2_local = coords[idx2]

        min_row = props["bbox-0"][idx]
        min_col = props["bbox-1"][idx]

        pt1_global_y = pt1_local[0] + min_row - 2
        pt1_global_x = pt1_local[1] + min_col - 2
        pt2_global_y = pt2_local[0] + min_row - 2
        pt2_global_x = pt2_local[1] + min_col - 2

        ax.plot(
            [pt1_global_x, pt2_global_x],
            [pt1_global_y, pt2_global_y],
            color="cyan",
            linestyle="--",
            linewidth=1.2,
            marker="o",
            markersize=3,
        )
        # --- end ---

    return fig


def export_to_csv(output_file, data, spacing=None):
    """
    Write data to csv.

    The function will use human-readable column names and convert values from pixels to
    microns (if spacing is given).

    Parameters
    ----------
    output_file : Path
        Path to output file
    data : DataFrame
        Output from regionprops_table, converted into a DataFrame
    spacing : float, optional
        Scaling in microns per pixel, by default None. If None, the values in pixels
        will be returned.
    """

    # Define human-readable headers for the CSV export
    header_map = {
        "label": "Object ID",
        "area": "Area (px)",
        "area_microns": "Area (micron2)",
        "feret_diameter_max": "Feret diameter (px)",
        "feret_diameter_max_microns": "Feret diameter (micron)",
    }

    # Convert to microns if spacing exists
    if spacing:
        if "area" in data.columns:
            data["area_microns"] = data["area"] * (spacing**2)

        if "feret_diameter_max" in data.columns:
            data["feret_diameter_max_microns"] = data["feret_diameter_max"] * spacing

    # Rename the columns
    data = data.rename(columns=header_map)

    # Move the label column to the left
    leading_cols = ["Object ID"]
    if "Image" in data.columns:
        leading_cols = ["Image"] + leading_cols

    data = data[leading_cols + [col for col in data.columns if col not in leading_cols]]

    data.to_csv(output_file, index=False)


def segment_cells(
    image, threshold=0.99, min_size=75, segment_inner=False, debug_plot=False
):
    """
    Segment organoids.

    This function uses intensity thresholding to identify organoids. The threshold is
    determined by threshold * max(image). In other words, we assume that the organoids
    are dark against a bright (white) background.

    The watershed parameters are calculated automatically. The absolute threshold is
    determined by the size of the objects. The parameter min_size is used to control how
    close the seed points are to each other, which affects the minimum size each object
    can be.

    Additionally, the function also tries to remove outlier objects which are too big
    (undersegmented) or too small (oversegmented) compared to the average size of the
    objects. This is defined as objects which have a size greater than mean + 7 * the stdev of
    the size.

    Parameters
    ----------
    image : ndarray
        Input image. If image is RGB, it will be converted to gray.
    threshold : float, optional
        Threshold factor, by default 0.99. A higher threshold factor will result in more
        pixels being labeled True.
    min_size : float, optional
        Minimum distance between watershed seed points, which translates to minimum
        object size. If processing a large batch, set this to the smallest expected
        object size.
    segment_inner : bool, optional
        If True, will also segment the internal dark region of the organoid. By default False.
    debug_plot : bool, optional
        If True, will generate plots to optimize segmentation parameters, by default False

    Returns
    -------
    label : ndarray
        Object labels
    inner_labels : ndarray, optional
        Labels of the inner regions. The label values match the object labels. This is
        only returned if segment_inner is True.
    """

    # Check if image is RGB
    if len(image.shape) == 2:
        pass  # Image is grayscale
    elif len(image.shape) == 3:
        if image.shape[-1] == 3:
            image = skimage.color.rgb2gray(image)
        else:
            raise ValueError(
                f"Expected the image to be RGB. Instead it has {image.shape[-1]} channels."
            )
    else:
        raise ValueError(
            f"Expected image to have shape H x W (grayscale) or H x W x 3 (RGB). Instead its shape was {image.shape}"
        )

    # Threshold the image as a percentage of the maximum
    mask = image < (threshold * np.max(image))

    mask = skimage.morphology.opening(mask, skimage.morphology.disk(30))
    mask = skimage.morphology.remove_small_holes(mask, max_size=200)

    # Watershed
    distance = ndi.distance_transform_edt(mask)
    coords = skimage.feature.peak_local_max(
        distance,
        footprint=np.ones((3, 3)),
        labels=mask,
        threshold_abs=(0.5 * np.max(distance)),
        min_distance=min_size,
    )

    mask_marker = np.zeros(distance.shape, dtype=bool)
    mask_marker[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_marker)
    labels = skimage.segmentation.watershed(-distance, markers, mask=mask)

    labels = skimage.segmentation.clear_border(labels)

    # Remove objects which are too big/small
    props = skimage.measure.regionprops_table(labels, properties=("area",))
    mean_area = np.mean(props["area"])
    stdev_area = np.std(props["area"])

    min_area = mean_area - (7 * stdev_area)

    labels = skimage.morphology.remove_small_objects(labels, max_size=min_area)

    # Make debug plots
    if debug_plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(image, cmap="gray")
        axes[0, 0].set_title("Input image (grayscale)")

        ov_mask = skimage.segmentation.mark_boundaries(
            image, mask, mode="thick", color=(0, 1, 0)
        )

        axes[0, 1].imshow(ov_mask)
        axes[0, 1].set_title("Mask overlay")

        ov_labels = skimage.segmentation.mark_boundaries(
            image, labels, mode="thick", color=(1, 0, 1)
        )
        axes[1, 0].imshow(ov_labels)
        axes[1, 0].set_title("Label overlay")

        plt.show()

    if segment_inner:
        # Do a global threshold to determine dark/light threshold
        thresh_cell = skimage.filters.threshold_otsu(image[labels > 0])

        inner_cell_mask = image >= thresh_cell

        inner_cell_labels = labels.copy()
        inner_cell_labels[~inner_cell_mask] = 0

        return (labels, inner_cell_labels)

    else:
        return labels


def segment_cells_dark(image, thresh=0.99):
    # This is used for the "ES" cells

    mask = image > (thresh * np.max(image))

    # plt.imshow(mask)
    # plt.show()

    # exit()

    # mask = skimage.morphology.opening(mask, skimage.morphology.disk(30))
    mask = skimage.morphology.remove_small_holes(mask, max_size=100000)
    mask = skimage.morphology.opening(mask, skimage.morphology.disk(5))

    # Watershed
    distance = ndi.distance_transform_edt(mask)
    coords = skimage.feature.peak_local_max(
        distance,
        footprint=np.ones((3, 3)),
        labels=mask,
        threshold_abs=(0.3 * np.max(distance)),
        min_distance=75,
    )

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

    # output_test = imoverlay(image, labels, color=[0, 1, 0, 0.4], plot_outlines=False)

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
            curr_slice = (
                (image_A - np.min(image_A)) / (np.max(image_A) - np.min(image_A)) * 255
            )
        else:
            curr_slice = image_A[:, :, c]

        if len(color) < 4:
            alpha = 1
        else:
            alpha = color[3]

        curr_slice[image_B] = (color[c] * 255 * alpha) + (
            (1 - alpha) * curr_slice[image_B]
        )
        image_out[:, :, c] = curr_slice

    return image_out


def dev_test_cp(input_path, output_dir):

    if isinstance(input_path, str):
        input_path = Path(input_path)

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if input_path.is_file():
        file_list = [input_path.resolve]
    else:
        file_list = list(input_path.glob("*.tif"))

    # imgs should be a list of images
    imgs = [skimage.io.imread(f) for f in file_list]

    # img = skimage.io.imread(input_path)

    model = models.CellposeModel(gpu=True)  # Runs cellpose sam
    masks, _, _ = model.eval(imgs, diameter=50)

    # Watershed and save the images
    for idx, mask in enumerate(masks):
        distance = ndi.distance_transform_edt(mask)
        coords = skimage.feature.peak_local_max(
            distance,
            footprint=np.ones((3, 3)),
            labels=mask,
            threshold_abs=(0.3 * np.max(distance)),
            min_distance=75,
        )

        mask_marker = np.zeros(distance.shape, dtype=bool)
        mask_marker[tuple(coords.T)] = True

        markers, _ = ndi.label(mask_marker)
        labels = skimage.segmentation.watershed(-distance, markers, mask=mask)
        labels = skimage.segmentation.clear_border(labels)

        output_test = imoverlay(
            imgs[idx], labels, color=[0, 1, 0, 0.4], plot_outlines=False
        )

        fn = file_list[idx].stem

        skimage.io.imsave(output_dir / (fn + ".png"), output_test)

        cell_props = skimage.measure.regionprops(labels)

        with open(os.path.join(output_dir, fn + ".csv"), "w", newline="") as file:
            writer = csv.writer(file, delimiter=",")

            # Write CSV headers
            writer.writerow(["Cell", "Label", "Total area (px)"])

            ctr = 0
            for p in cell_props:
                writer.writerow([ctr + 1, p.label, p.area])
                ctr += 1


if __name__ == "__main__":
    # TODO: CLI
    pass
