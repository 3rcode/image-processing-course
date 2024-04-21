import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_img(img_path):
    """
    Read grayscale image
    Inputs:
    img_path: str: image path
    Returns:
    img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    assert filter_size % 2 == 1, "Filter size must be odd number"
    assert filter_size <= min(img.shape), "Filter size must not be too large"
    s = filter_size // 2
    pad_top = np.tile(img[0, :], (s, 1))
    pad_bot = np.tile(img[-1, :], (s, 1))
    img = np.concatenate([pad_top, img, pad_bot], axis=0)
    pad_left = np.tile(img[:, 0], (s, 1)).T
    pad_right = np.tile(img[:, -1], (s, 1)).T
    img = np.concatenate([pad_left, img, pad_right], axis=1)
    return img


def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    convolve = np.zeros((img.shape))
    img = padding_img(img, filter_size)
    filter = np.ones((filter_size, filter_size)) / (filter_size**2)
    s = filter_size // 2
    x, y = img.shape
    for v in range(s, x - s):
        for h in range(s, y - s):  #
            area = img[(v - s) : (v + s + 1), (h - s) : (h + s + 1)]
            convolve[v - s, h - s] = np.sum(np.multiply(filter, area))
    return convolve


def median_filter(img, filter_size=3):
    """
    Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        smoothed_img: cv2 image: the smoothed image with median filter.
    """
    median = np.zeros((img.shape))
    img = padding_img(img, filter_size)
    s = filter_size // 2
    x, y = img.shape
    for v in range(s, x - s):
        for h in range(s, y - s):
            area = img[(v - s) : (v + s + 1), (h - s) : (h + s + 1)]
            median[v - s, h - s] = np.median(area)
    return median


def psnr(gt_img, smooth_img):
    """
    Calculate the PSNR metric
    Inputs:
        gt_img: cv2 image: groundtruth image
        smooth_img: cv2 image: smoothed image
    Outputs:
        psnr_score: PSNR score
    """
    try:
        gt_img = np.array(gt_img)
        smooth_img = np.array(smooth_img)
    except Exception:
        raise ValueError("Input must be 2D array like format")
    max_possible_value = 255
    mse = np.mean((gt_img - smooth_img) ** 2)
    return 10 * np.log10(max_possible_value**2 / mse)


def show_res(before_img, after_img):
    """
    Show the original image and the corresponding smooth image
    Inputs:
        before_img: cv2: image before smoothing
        after_img: cv2: corresponding smoothed image
    Return:
        None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap="gray")
    plt.title("Before")

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap="gray")
    plt.title("After")
    plt.show()


if __name__ == "__main__":
    img_noise = "./images/noise.png"  # <- need to specify the path to the noise image
    img_gt = "./images/ori_img.png"  # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print("PSNR score of mean filter: ", psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print("PSNR score of median filter: ", psnr(img, median_smoothed_img))
