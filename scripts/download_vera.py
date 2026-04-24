import os
import zipfile
import argparse
from huggingface_hub import hf_hub_download
from src.config import RAW_DATA_DIR

def download_and_extract(repo_id, filename, token=None):
    """
    Downloads a dataset zip from a given Hugging Face repository 
    and extracts it into the data/raw/vera directory.
    """
    print(f"Connecting to Hugging Face Repo: {repo_id}...")
    
    # Download the zip file
    try:
        zip_path = hf_hub_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            filename=filename, 
            token=token,
            local_dir=RAW_DATA_DIR
        )
        print(f"✅ Successfully downloaded to: {zip_path}")
    except Exception as e:
        print(f"❌ Failed to download from Hugging Face: {e}")
        print("Note: If the repository is private, ensure you provided a valid HF_TOKEN.")
        return

    # Unzip the downloaded file
    extract_dir = os.path.join(RAW_DATA_DIR, "vera")
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting {zip_path} into {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Extraction complete! Files are ready in {extract_dir}")
        
        # Optional: delete the zip file to save Vast.ai storage space
        os.remove(zip_path)
        print("🗑️ Removed the original zip file to conserve GPU disk space.")
        
    except zipfile.BadZipFile:
        print("❌ Error: The downloaded file is not a valid zip archive.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download VERA dataset from Hugging Face")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face Repository ID (e.g., 'SumitSat/vera-ransomware')")
    parser.add_argument("--file", type=str, required=True, help="Name of the zip file in the repo (e.g., 'dataset.zip')")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Read Access Token (Required if repo is private)")
    
    args = parser.parse_args()
    
    # We can also read the token from environment variables
    hf_token = args.token or os.environ.get("HF_TOKEN")
    
    download_and_extract(args.repo, args.file, hf_token)
