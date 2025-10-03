"""
Data augmentation utilities for improving model performance
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from config import *

class DataAugmentor:
    def __init__(self):
        self.keras_augmentor = None
        self.albumentations_augmentor = None
        self.setup_augmentations()
    
    def setup_augmentations(self):
        """Setup different augmentation pipelines"""
        
        # Keras ImageDataGenerator
        self.keras_augmentor = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        # Albumentations pipeline (more advanced)
        try:
            self.albumentations_augmentor = A.Compose([
                A.RandomRotate90(p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.PiecewiseAffine(p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
            ])
        except ImportError:
            print("Albumentations not installed. Using basic augmentation only.")
            self.albumentations_augmentor = None
    
    def augment_image_albumentations(self, image):
        """Augment single image using Albumentations"""
        if self.albumentations_augmentor is None:
            return image
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        # Apply augmentation
        augmented = self.albumentations_augmentor(image=image_uint8)
        augmented_image = augmented['image']
        
        # Convert back to float32
        return augmented_image.astype(np.float32) / 255.0
    
    def create_augmented_dataset(self, X, y, augmentation_factor=2):
        """Create augmented dataset"""
        print(f"Creating augmented dataset with factor {augmentation_factor}...")
        
        X_augmented = []
        y_augmented = []
        
        # Add original data
        X_augmented.extend(X)
        y_augmented.extend(y)
        
        # Generate augmented samples
        for i in range(len(X)):
            for _ in range(augmentation_factor - 1):  # -1 because we already have original
                if self.albumentations_augmentor is not None:
                    aug_image = self.augment_image_albumentations(X[i])
                    X_augmented.append(aug_image)
                    y_augmented.append(y[i])
                
                if i % 1000 == 0:
                    print(f"Augmented {i}/{len(X)} images")
        
        print(f"Augmentation complete. Dataset size: {len(X)} -> {len(X_augmented)}")
        
        return np.array(X_augmented), np.array(y_augmented)
    
    def get_keras_generator(self, X, y, batch_size=BATCH_SIZE, shuffle=True):
        """Get Keras data generator with augmentation"""
        return self.keras_augmentor.flow(X, y, batch_size=batch_size, shuffle=shuffle)
    
    def visualize_augmentations(self, image, num_augmentations=8):
        """Visualize different augmentations of a single image"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Generate augmented versions
        for i in range(1, 9):
            row = i // 3
            col = i % 3
            
            if self.albumentations_augmentor is not None:
                aug_image = self.augment_image_albumentations(image)
            else:
                # Fallback to simple augmentation
                aug_image = image
            
            axes[row, col].imshow(aug_image)
            axes[row, col].set_title(f'Augmented {i}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(LOGS_DIR, 'augmentation_examples.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Test data augmentation"""
    augmentor = DataAugmentor()
    
    # Create dummy image for testing
    dummy_image = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Test augmentation
    if augmentor.albumentations_augmentor is not None:
        augmented = augmentor.augment_image_albumentations(dummy_image)
        print(f"Augmentation test successful. Original shape: {dummy_image.shape}, Augmented shape: {augmented.shape}")
    else:
        print("Albumentations not available, using basic augmentation only")
    
    print("Data augmentation module initialized successfully!")

if __name__ == "__main__":
    main()