from PIL import Image

# Load the image
input_path = "your_figure.png"
output_path = "your_figure_cropped.png"

top = 500
left = 360
right = 330
bot = 525

with Image.open(input_path) as img:
    width, height = img.size
    print(width, height)

    # Define the crop box (left, upper, right, lower)
    crop_box = (left, top, width - right, height - bot)

    # Perform the crop
    cropped_img = img.crop(crop_box)

    # Save the result
    cropped_img.save(output_path)

print(f"Cropped image saved to {output_path}")
