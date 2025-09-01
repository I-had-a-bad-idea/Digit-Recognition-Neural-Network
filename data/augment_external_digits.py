import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageChops, ImageOps
import random
from scipy.ndimage import gaussian_filter, map_coordinates


class Augmented_digits_generator:
    def preprocess_digit(self, arr):
        if isinstance(arr, Image.Image):
            arr = np.array(arr.convert("L"))

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        arr = (arr > 128).astype(np.uint8) * 255

        coords = np.argwhere(arr > 0)
        if coords.shape[0] == 0:
            return np.zeros(784, dtype=np.float32)

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = arr[y0:y1, x0:x1]

        h, w = cropped.shape
        scale = 20.0 / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        pil_img = Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS)
        digit = np.array(pil_img)

        coords = np.argwhere(digit > 0)
        y_center, x_center = coords.mean(axis=0)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        y_offset = int(round(14 - y_center))
        x_offset = int(round(14 - x_center))
        y_start, x_start = max(0, y_offset), max(0, x_offset)
        y_end, x_end = min(28, y_offset + new_h), min(28, x_offset + new_w)
        canvas[y_start:y_end, x_start:x_end] = digit[0:(y_end-y_start), 0:(x_end-x_start)]

        return (canvas.astype(np.float32) / 255.0).reshape(784)

    def elastic_transform(self, image, alpha: float = 36.0, sigma: float = 6.0):
        arr = np.array(image)

        random_state = np.random.RandomState(None)
        shape = arr.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

        distorted = map_coordinates(arr, indices, order=1, mode='reflect').reshape(shape)
        return Image.fromarray(distorted.astype(np.uint8))

    def generate_augmented_data(self, input_dir, samples_per_image=100):
        images, labels = [] , []

        def augment_image(img):
            # Elastic distortion (applied with 30% chance)
            if random.random() < 0.3:
                img = self.elastic_transform(img, alpha=random.uniform(30, 40), sigma=random.uniform(5, 7))

            # Rotation
            if random.random() < 0.7:
                angle = random.uniform(-20, 20)
                img = img.rotate(angle, fillcolor=0)

            # Shear
            if random.random() < 0.5:
                shear = random.uniform(-0.3, 0.3)
                img = img.transform(img.size, Image.Transform.AFFINE,
                                    (1, shear, 0, shear, 1, 0),
                                    fillcolor=0)

            # Shifts
            if random.random() < 0.7:
                dx, dy = random.randint(-3, 3), random.randint(-3, 3)
                img = ImageChops.offset(img, dx, dy)

            # Scaling
            if random.random() < 0.5:
                scale = random.uniform(0.7, 1.3)
                new_size = int(img.size[0] * scale), int(img.size[1] * scale)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                canvas = Image.new("L", (28, 28), color=0)
                x_off = (28 - new_size[0]) // 2
                y_off = (28 - new_size[1]) // 2
                canvas.paste(img, (x_off, y_off))
                img = canvas

            # Contrast & brightness
            if random.random() < 0.5:
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

            # Stroke thickness
            if random.random() < 0.3:
                img = img.filter(ImageFilter.MaxFilter(3))
            elif random.random() < 0.3:
                img = img.filter(ImageFilter.MinFilter(3))

            # Blur / noise
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
            if random.random() < 0.2:
                arr = np.array(img).astype(np.float32)
                noise = np.random.normal(0, 25, arr.shape)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)

            # Occasionally invert
            if random.random() < 0.1:
                img = ImageOps.invert(img)

            return img

        for file in os.listdir(input_dir):
            if not file.endswith(".png"):
                continue
            label_str = os.path.splitext(file)[0][0]
            if not label_str.isdigit():
                continue
            label = int(label_str)

            base_img = Image.open(os.path.join(input_dir, file)).convert("L")

            for _ in range(samples_per_image):
                img = augment_image(base_img)
                arr = self.preprocess_digit(img)
                images.append(arr)
                labels.append(label)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
