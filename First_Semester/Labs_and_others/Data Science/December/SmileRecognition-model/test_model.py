"""
Test the trained smile recognition model on your own images.
Place your images in the 'test_images' folder and run this script.
"""

import os
import numpy as np
from model import SmileRecognitionModel, ImagePreprocessor

def test_on_images():
    """Test the model on images in the test_images folder."""
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "test_images")
    weights_path = os.path.join(script_dir, "smile_model_weights.npz")
    
    # Check if test folder exists
    if not os.path.exists(test_dir):
        print(f"Error: test_images folder not found!")
        print(f"Please create: {test_dir}")
        return
    
    # Check if model weights exist
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found!")
        print(f"Please run 'python model.py' first to train the model.")
        return
    
    # Get image files
    supported_formats = ('.png', '.jpg', '.jpeg', '.ppm', '.pgm', '.bmp')
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(supported_formats)]
    
    if len(image_files) == 0:
        print("=" * 60)
        print("No images found in test_images folder!")
        print("=" * 60)
        print(f"\nPlease add images to: {test_dir}")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return
    
    print("=" * 60)
    print("Smile Recognition - Testing on Your Images")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Loading model...")
    model = SmileRecognitionModel(
        image_size=(64, 64),
        feature_type='hog',
        hidden_layers=[128, 64]
    )
    
    # Build model with correct input size (HOG features for 64x64 image)
    model.build_model(1764)  # HOG feature size
    model.load(weights_path)
    print("   Model loaded successfully!")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(64, 64))
    
    # Test each image
    print(f"\n2. Testing {len(image_files)} images...")
    print("-" * 60)
    print(f"{'Image':<40} {'Prediction':<12} {'Confidence':<10}")
    print("-" * 60)
    
    results = []
    
    for filename in sorted(image_files):
        image_path = os.path.join(test_dir, filename)
        
        try:
            # Load and predict
            image = preprocessor.load_image(image_path)
            probability = model.predict_image(image)
            
            # Determine prediction
            is_smile = probability >= 0.5
            label = "SMILE 😊" if is_smile else "NO SMILE 😐"
            confidence = probability if is_smile else (1 - probability)
            
            results.append({
                'filename': filename,
                'probability': probability,
                'is_smile': is_smile,
                'confidence': confidence
            })
            
            # Print result
            print(f"{filename:<40} {label:<12} {confidence*100:>6.1f}%")
            
        except Exception as e:
            print(f"{filename:<40} ERROR: {str(e)[:30]}")
    
    # Summary
    print("-" * 60)
    
    if results:
        n_smiles = sum(1 for r in results if r['is_smile'])
        n_no_smiles = len(results) - n_smiles
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"\nSummary:")
        print(f"   Total images:      {len(results)}")
        print(f"   Detected smiles:   {n_smiles}")
        print(f"   No smiles:         {n_no_smiles}")
        print(f"   Avg confidence:    {avg_confidence*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    test_on_images()
