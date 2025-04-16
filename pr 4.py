import os
import numpy as np
from PIL import Image
import qrcode
import cv2
import matplotlib.pyplot as plt


# Function to create a folder if it doesn't exist and save the image
def create_and_save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, filename)
    image.save(path)
    return path


# Function to display image using matplotlib
def show_image(image_path, title):
    img = Image.open(image_path)
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


# Function to generate a QR code from a URL or link
def generate_qr_code(input_string, folder="encoded_images"):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=10, border=4)
    qr.add_data(input_string)  # You can input a URL here as well
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    qr_image_path = create_and_save_image(img, folder, "qr_code.png")
    show_image(qr_image_path, "QR Code")
    return img


# Function to apply visual cryptography (2,2 scheme)
def apply_visual_cryptography(qr_image):
    # Convert QR image to binary (black and white)
    qr_image = qr_image.convert("1")
    qr_array = np.array(qr_image)

    # Create two empty shares
    share1 = np.zeros_like(qr_array)
    share2 = np.zeros_like(qr_array)

    # (2, 2) Scheme: Split the binary QR into two shares
    for i in range(qr_array.shape[0]):
        for j in range(qr_array.shape[1]):
            if qr_array[i][j] == 0:
                # 00 -> 0, 0; 01 -> 0, 1; 10 -> 1, 0; 11 -> 1, 1
                share1[i, j] = 0
                share2[i, j] = 0
            else:
                share1[i, j] = 1
                share2[i, j] = 1

    # Convert to image and save
    share1_img = Image.fromarray(share1 * 255)  # Convert to black and white image
    share2_img = Image.fromarray(share2 * 255)  # Convert to black and white image

    share1_path = create_and_save_image(share1_img, "encoded_images", "share1.png")
    share2_path = create_and_save_image(share2_img, "encoded_images", "share2.png")

    show_image(share1_path, "Share 1")
    show_image(share2_path, "Share 2")

    return share1_img, share2_img


# Function to decode the shares and reconstruct the QR code
def decode_visual_cryptography(share1_path, share2_path, folder="decoded_images"):
    share1 = Image.open(share1_path).convert("1")
    share2 = Image.open(share2_path).convert("1")

    # Combine shares
    share1_array = np.array(share1)
    share2_array = np.array(share2)

    # Reconstruct the original QR
    reconstructed = np.bitwise_or(share1_array, share2_array)

    # Convert to image
    reconstructed_image = Image.fromarray(reconstructed * 255)

    # Save and display
    decoded_image_path = create_and_save_image(reconstructed_image, folder, "decoded_qr_code.png")
    show_image(decoded_image_path, "Decoded QR Code")

    # Decode the QR code
    decoded_qr = cv2.imread(decoded_image_path, cv2.IMREAD_GRAYSCALE)
    decoded_data, points, _ = cv2.QRCodeDetector().detectAndDecode(decoded_qr)
    print("Decoded string from QR code:", decoded_data)

    return decoded_data


# Function to scan QR codes from an image
def scan_qr_from_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize QR Code detector
    qr_decoder = cv2.QRCodeDetector()

    # Detect and decode QR code
    decoded_data, points, _ = qr_decoder.detectAndDecode(img)

    if decoded_data:
        print(f"Decoded string from image: {decoded_data}")
    else:
        print("No QR code found in the image.")

    return decoded_data


# Function to scan QR codes from a video
def scan_qr_from_video(video_source=0):
    # Initialize video capture (0 for default camera, or path to video file)
    cap = cv2.VideoCapture(video_source)

    # Initialize QR Code detector
    qr_decoder = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and decode QR code in the frame
        decoded_data, points, _ = qr_decoder.detectAndDecode(frame)

        if decoded_data:
            print(f"Decoded string from video: {decoded_data}")
            break

        # Show the frame with QR code detection (if any)
        cv2.imshow("QR Code Scanner", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main function
if __name__ == "__main__":
    input_string = input("Enter a URL or link to encode into a QR code: ")
    input_string2 = input("Enter string to encode into a QR code: ")

    # Generate QR code with the URL or link
    qr_image = generate_qr_code(input_string)
    qr_image = generate_qr_code(input_string2)

    # Apply visual cryptography (2,2 scheme)
    share1_img, share2_img = apply_visual_cryptography(qr_image)

    # Decode visual cryptography (reconstruct QR code and decode)
    decoded_string = decode_visual_cryptography("encoded_images/share1.png", "encoded_images/share2.png")
    print(f"Decoded string: {decoded_string}")

