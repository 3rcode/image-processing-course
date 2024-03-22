import cv2 
import matplotlib.pyplot as plt
import numpy as np

# Load an image from file as function
def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file, using OpenCV
    """
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    

# Display an image as function
def display_image(image: np.ndarray, title: str="Image"):
    """
    Display an image using matplotlib. Rembember to use plt.show() to display the image
    """
    if image.ndim == 3 and image.shape[2] == 3:
        plt.imshow(image[..., ::-1], vmin=0, vmax=255)
    elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.show()

# Grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    return np.dot(image[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)



# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path, image)

# Flip an image as function 
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, ::-1, :]
    elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return image[:, ::-1]


# rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_img = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_img


if __name__ == "__main__":
    # Load an image from file
    img = load_image("./images/uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "./images/lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Saved the flipped grayscale image
    save_image(img_gray_flipped, "./images/lena_gray_flipped.jpg")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "./images/lena_gray_rotated.jpg")
