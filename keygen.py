import os
ssh_dir = os.path.expanduser("~/.ssh")
os.makedirs(ssh_dir, exist_ok=True)
key_path = os.path.join(ssh_dir, "id_rsa_vast")

# Generate the key if it doesn't exist
if not os.path.exists(key_path):
    os.system(f'ssh-keygen -t rsa -b 4096 -f "{key_path}" -q -N ""')

# Print the public key
with open(f"{key_path}.pub", "r") as f:
    print("=== YOUR PUBLIC SSH KEY ===")
    print(f.read().strip())
    print("===========================")
