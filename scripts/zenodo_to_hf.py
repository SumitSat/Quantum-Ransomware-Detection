import os
import requests
import zipfile
import sys
from huggingface_hub import login, upload_folder
from pathlib import Path

def setup_huggingface_repo(repo_id, token):
    try:
        from huggingface_hub import create_repo
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True, private=False)
        print(f"Verified/Created repository: {repo_id}")
    except Exception as e:
        print(f"Error checking repo: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python zenodo_to_hf.py <HF_TOKEN>")
        sys.exit(1)
        
    token = sys.argv[1]
    repo_id = "Shashwat1055/Quantum-Ransomware_DB"
    
    print("--- 1. Fetching Zenodo record 17904705 ---")
    r = requests.get("https://zenodo.org/api/records/17904705")
    data = r.json()
    files = data.get("files", [])
    
    os.makedirs("vera_dataset", exist_ok=True)
    os.chdir("vera_dataset")
    
    for f in files:
        link_dict = f.get("links", {})
        url = link_dict.get("self", "")
        if "api/" in url and not url.endswith("content"):
            url = url + "/content"
        elif not url:
            url = f"https://zenodo.org/records/17904705/files/{filename}?download=1"
        filename = f["key"]
        print(f"\n=> Downloading {filename} from Zenodo...")
        
        # Using wget because it handles massive 10GB+ files seamlessly and gives progress
        os.system(f"wget -q --show-progress -O {filename} {url}")
        
        if filename.endswith(".zip"):
            print(f"=> Extracting {filename}...")
            try:
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(".")
                os.remove(filename)  # Delete zip after extraction to save GPU disk
                print(f"Removed {filename}")
            except Exception as e:
                print(f"Error unzipping {filename}: {e}")
                
    # Upload to HF
    print("\n--- 2. Uploading securely to Hugging Face ---")
    login(token=token)
    setup_huggingface_repo(repo_id, token)
    
    # We upload the current directory to HF
    print(f"Pushing extracted dataset directly to {repo_id}...")
    upload_folder(
        folder_path=".", 
        repo_id=repo_id, 
        repo_type="dataset"
    )
    print("\n✅ Dataset successfully migrated from Zenodo to Hugging Face!")

if __name__ == "__main__":
    main()
