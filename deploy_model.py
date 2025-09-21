
import os
import shutil
import json

def deploy_model():
    """
    Automates the deployment of a trained PyTorch model to the Flask backend.
    """
    print("🚀 Starting model deployment...")

    # --- 1. Define Paths ---
    source_dir = './models/'
    destination_dir = './backend/models/'
    model_file = 'handicraft_cnn.pth'
    class_indices_file = 'class_indices.json'

    source_model_path = os.path.join(source_dir, model_file)
    source_indices_path = os.path.join(source_dir, class_indices_file)

    print(f"   - Source directory:      {os.path.abspath(source_dir)}")
    print(f"   - Destination directory: {os.path.abspath(destination_dir)}")

    # --- 2. Verify Source Files ---
    print("\n🔍 Verifying that source files exist...")
    if not os.path.exists(source_model_path):
        print(f"❌ Error: Model file not found at '{source_model_path}'.")
        print("   Please run the training script (train_model.py) to generate the model.")
        return

    if not os.path.exists(source_indices_path):
        print(f"❌ Error: Class indices file not found at '{source_indices_path}'.")
        print("   Please run the training script (train_model.py) to generate the class indices.")
        return

    print("✅ Source files verified successfully.")

    # --- 3. Ensure Destination Exists ---
    print(f"\n📁 Checking destination directory...")
    if not os.path.exists(destination_dir):
        print(f"   - Destination directory not found. Creating it now...")
        os.makedirs(destination_dir)
        print("   - Directory created.")
    else:
        print("   - Destination directory already exists.")

    # --- 4. Copy Files ---
    print("\n🚚 Copying model files to the backend...")
    
    # Copy the model file
    dest_model_path = os.path.join(destination_dir, model_file)
    shutil.copy2(source_model_path, dest_model_path)
    print(f"   - Copied '{model_file}' to '{dest_model_path}'")

    # Copy the class indices file
    dest_indices_path = os.path.join(destination_dir, class_indices_file)
    shutil.copy2(source_indices_path, dest_indices_path)
    print(f"   - Copied '{class_indices_file}' to '{dest_indices_path}'")

    # --- 5. Provide Feedback and Final Reminder ---
    print("\n🎉 Deployment complete! Your model is now ready for the backend.")
    print("=" * 60)
    print("🔔 IMPORTANT REMINDER:")
    print("   You must restart your Flask backend server for the new model to be loaded.")
    print("   If your server is running, stop it (Ctrl+C) and start it again.")
    print("=" * 60)

if __name__ == '__main__':
    deploy_model()
