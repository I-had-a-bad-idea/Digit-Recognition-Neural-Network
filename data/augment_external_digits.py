import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageChops
import random

class Augmented_digits_generator:
    def generate_augmented_data(self, input_dir, samples_per_image=100):
        images, labels = [], []

        def augment_image(img):
            # Random rotation ±20°
            if random.random() < 0.7:
                angle = random.uniform(-20, 20)
                img = img.rotate(angle, fillcolor=0)

            # Random shifts
            if random.random() < 0.7:
                dx, dy = random.randint(-3, 3), random.randint(-3, 3)
                img = ImageChops.offset(img, dx, dy)

            # Random contrast
            if random.random() < 0.5:
                factor = random.uniform(0.7, 1.3)
                img = ImageEnhance.Contrast(img).enhance(factor)

            # Random brightness
            if random.random() < 0.5:
                factor = random.uniform(0.7, 1.3)
                img = ImageEnhance.Brightness(img).enhance(factor)

            # Stroke thickness (dilate/erode effect)
            if random.random() < 0.5:
                img = img.filter(ImageFilter.MaxFilter(3))  # thicken
            elif random.random() < 0.5:
                img = img.filter(ImageFilter.MinFilter(3))  # thin

            return img

        # Iterate through labeled PNGs
        for file in os.listdir(input_dir):
            if not file.endswith(".png"):
                continue

            # Label = first character of filename (e.g., "2.png" -> 2)
            label_str = os.path.splitext(file)[0][0]
            if not label_str.isdigit():
                continue
            label = int(label_str)

            base_img = Image.open(os.path.join(input_dir, file)).convert("L")

            for _ in range(samples_per_image):
                img = augment_image(base_img)

                # Resize/center to 28x28
                arr = np.array(img)
                arr = (arr > 128).astype(np.uint8) * 255
                pil_img = Image.fromarray(arr).resize((28, 28), Image.Resampling.LANCZOS)

                arr = np.array(pil_img).astype(np.float32) / 255.0
                images.append(arr.flatten())
                labels.append(label)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)