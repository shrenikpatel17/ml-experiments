import os
import random
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# apply transformations to images
def apply_transformations(image, transformation_type):
    if transformation_type == "orientation":
        angle = random.choice([90, 180, 270])
        return image.rotate(angle)
    
    elif transformation_type == "flip":
        if random.random() > 0.5:
            return ImageOps.mirror(image)  
        return ImageOps.flip(image) 
    
    elif transformation_type == "quantity":
        width, height = image.size
        new_img = Image.new('RGB', (width * 2, height * 2))
        for i in range(2):
            for j in range(2):
                enhancer = ImageEnhance.Color(image)
                variation = enhancer.enhance(0.8 + random.random() * 0.4)
                new_img.paste(variation, (i * width, j * height))
        return new_img
    
    elif transformation_type == "position":
        width, height = image.size
        new_width = int(width * 1.5)
        new_height = int(height * 1.5)
        new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        x_offset = random.randint(0, new_width - width)
        y_offset = random.randint(0, new_height - height)
        new_img.paste(image, (x_offset, y_offset))
        return new_img
    
    elif transformation_type == "color":
        transforms_list = [
            ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5)),  
            ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3)), 
            ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3)) 
        ]
        return random.choice(transforms_list)
    
    elif transformation_type == "zoom":
        width, height = image.size
        crop_factor = random.uniform(0.6, 0.8)
        crop_width = int(width * crop_factor)
        crop_height = int(height * crop_factor)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height))
    
    elif transformation_type == "text":
        img_draw = ImageDraw.Draw(image)
        width, height = image.size
        text_options = ["Object", "Item", "Thing", "Sample"]
        text = random.choice(text_options)
        img_draw.text((width//4, height//4), text, fill=(255, 255, 255))
        return image
    
    return image

def create_transformed_dataset(input_dir='./selected_images', output_dir='./transformed_dataset'):
    os.makedirs(output_dir, exist_ok=True)
    
    transformations = [
        "orientation", "flip", "quantity", "position", 
        "color", "zoom", "text"
    ]
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.JPEG', '.png'))]
    
    print("Creating transformed dataset...")
    for img_file in tqdm(image_files):
        if img_file == 'metadata.csv' or img_file == 'samples.png':
            continue
            
        img_path = os.path.join(input_dir, img_file)
        try:
            original_img = Image.open(img_path).convert('RGB')
            
            original_path = os.path.join(output_dir, f"original_{img_file}")
            original_img.save(original_path)
            
            for transform_type in transformations:
                transformed_img = apply_transformations(original_img.copy(), transform_type)
                transformed_path = os.path.join(output_dir, f"{transform_type}_{img_file}")
                transformed_img.save(transformed_path)
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"Dataset created in {output_dir}")
    print(f"Each image has {len(transformations)} variations")
    return True

if __name__ == "__main__":
    success = create_transformed_dataset()
    if not success:
        print("Failed to create transformed dataset")
