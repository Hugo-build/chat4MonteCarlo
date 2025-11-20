#!/usr/bin/env python3
"""
Encrypt/Decrypt .env files for secure storage
Usage:
    python encrypt_env.py encrypt  # Creates .env.encrypted from .env
    python encrypt_env.py decrypt  # Creates .env from .env.encrypted
"""

import os
import sys
import getpass
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

def derive_key_from_password(password: str, salt: bytes) -> bytes:
    """Derive an encryption key from a password."""
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(password.encode())
    return Fernet.generate_key() if len(key) != 32 else key

def encrypt_env():
    """Encrypt .env file to .env.encrypted"""
    env_path = Path(".env")
    encrypted_path = Path(".env.encrypted")
    salt_path = Path(".env.salt")
    
    if not env_path.exists():
        print("❌ Error: .env file not found")
        return
    
    # Get password
    password = getpass.getpass("🔐 Enter encryption password: ")
    confirm = getpass.getpass("🔐 Confirm password: ")
    
    if password != confirm:
        print("❌ Passwords don't match!")
        return
    
    # Generate salt
    salt = os.urandom(16)
    salt_path.write_bytes(salt)
    
    # Derive key from password
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(password.encode())
    fernet = Fernet(Fernet.generate_key())  # Use proper Fernet key
    
    # For proper encryption, we need to use password-based encryption
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    import base64
    
    # Read .env content
    content = env_path.read_bytes()
    
    # Encrypt
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Pad content to block size
    pad_length = 16 - (len(content) % 16)
    padded_content = content + bytes([pad_length] * pad_length)
    
    encrypted = encryptor.update(padded_content) + encryptor.finalize()
    
    # Save encrypted file with IV prepended
    encrypted_path.write_bytes(iv + encrypted)
    
    print(f"✅ Encrypted .env saved to {encrypted_path}")
    print(f"🔑 Salt saved to {salt_path}")
    print(f"⚠️  Keep your password safe! You'll need it to decrypt.")
    print(f"💡 Add to .gitignore: .env (already there)")
    print(f"💡 You can commit: .env.encrypted and .env.salt")

def decrypt_env():
    """Decrypt .env.encrypted to .env"""
    encrypted_path = Path(".env.encrypted")
    salt_path = Path(".env.salt")
    env_path = Path(".env")
    
    if not encrypted_path.exists():
        print("❌ Error: .env.encrypted file not found")
        return
    
    if not salt_path.exists():
        print("❌ Error: .env.salt file not found")
        return
    
    # Get password
    password = getpass.getpass("🔐 Enter decryption password: ")
    
    # Read salt
    salt = salt_path.read_bytes()
    
    # Derive key from password
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(password.encode())
    
    # Read encrypted content
    encrypted_data = encrypted_path.read_bytes()
    iv = encrypted_data[:16]
    encrypted = encrypted_data[16:]
    
    # Decrypt
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    try:
        decrypted = decryptor.update(encrypted) + decryptor.finalize()
        
        # Remove padding
        pad_length = decrypted[-1]
        content = decrypted[:-pad_length]
        
        # Save .env file
        env_path.write_bytes(content)
        
        print(f"✅ Decrypted to {env_path}")
        print(f"🚀 You can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        print("⚠️  Check your password and try again")





def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python encrypt_env.py encrypt  # Encrypt .env file")
        print("  python encrypt_env.py decrypt  # Decrypt .env.encrypted file")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "encrypt":
        encrypt_env()
    elif command == "decrypt":
        decrypt_env()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'encrypt' or 'decrypt'")
        sys.exit(1)

if __name__ == "__main__":
    main()

