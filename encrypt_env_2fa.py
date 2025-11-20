#!/usr/bin/env python3
"""
Encrypt/Decrypt .env files with 2FA (Two-Factor Authentication)
Requires password + TOTP code (Google Authenticator, Authy, etc.)

Usage:
    python encrypt_env_2fa.py setup     # First-time setup: generates 2FA secret
    python encrypt_env_2fa.py encrypt   # Encrypt .env with password + 2FA
    python encrypt_env_2fa.py decrypt   # Decrypt .env.encrypted with password + 2FA
    python encrypt_env_2fa.py verify    # Test your 2FA code
"""

import os
import sys
import getpass
import qrcode
from pathlib import Path
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pyotp
import time

# File paths
ENV_FILE = Path(".env")
ENCRYPTED_FILE = Path(".env.encrypted")
SALT_FILE = Path(".env.salt")
TOTP_SECRET_FILE = Path(".env.2fa.secret")  # Encrypted TOTP secret


def setup_2fa():
    """Initial setup: Generate 2FA secret and show QR code."""
    print("\n" + "="*70)
    print("🔐 2FA SETUP - Two-Factor Authentication")
    print("="*70 + "\n")
    
    if TOTP_SECRET_FILE.exists():
        response = input("⚠️  2FA already set up. Reset? (yes/no): ")
        if response.lower() != "yes":
            print("Setup cancelled.")
            return
    
    # Generate TOTP secret
    totp_secret = pyotp.random_base32()
    
    # Create TOTP URI for QR code
    project_name = "pySMC"
    totp_uri = pyotp.totp.TOTP(totp_secret).provisioning_uri(
        name=f"{project_name}@env",
        issuer_name="pySMC Environment"
    )
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    
    print("📱 Scan this QR code with your authenticator app:")
    print("   (Google Authenticator, Authy, 1Password, etc.)\n")
    
    # Print QR code to terminal
    qr.print_ascii(invert=True)
    
    print("\n" + "-"*70)
    print("📋 Manual Entry (if QR doesn't work):")
    print(f"   Secret Key: {totp_secret}")
    print(f"   Account: {project_name}@env")
    print(f"   Type: Time-based (TOTP)")
    print("-"*70 + "\n")
    
    # Verify it works
    totp = pyotp.TOTP(totp_secret)
    while True:
        code = input("🔢 Enter the 6-digit code from your app to verify: ").strip()
        if totp.verify(code, valid_window=1):
            print("✅ 2FA verified successfully!")
            break
        else:
            print("❌ Invalid code. Try again.")
    
    # Encrypt the TOTP secret with a password
    password = getpass.getpass("\n🔑 Set a master password to protect 2FA: ")
    confirm = getpass.getpass("🔑 Confirm password: ")
    
    if password != confirm:
        print("❌ Passwords don't match!")
        return
    
    # Derive encryption key from password
    salt = os.urandom(16)
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(password.encode())
    
    # Encrypt TOTP secret
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Pad secret
    secret_bytes = totp_secret.encode()
    pad_length = 16 - (len(secret_bytes) % 16)
    padded_secret = secret_bytes + bytes([pad_length] * pad_length)
    
    encrypted_secret = encryptor.update(padded_secret) + encryptor.finalize()
    
    # Save encrypted secret with salt and IV
    TOTP_SECRET_FILE.write_bytes(salt + iv + encrypted_secret)
    
    print(f"\n✅ 2FA setup complete!")
    print(f"📁 Secret saved to: {TOTP_SECRET_FILE}")
    print(f"⚠️  Keep this file safe - you'll need it to decrypt!\n")
    print("Next steps:")
    print("  1. python encrypt_env_2fa.py encrypt  # Encrypt your .env")
    print("  2. python encrypt_env_2fa.py decrypt  # Decrypt when needed")


def load_totp_secret(password: str) -> str:
    """Load and decrypt TOTP secret."""
    if not TOTP_SECRET_FILE.exists():
        raise FileNotFoundError(
            "2FA not set up. Run: python encrypt_env_2fa.py setup"
        )
    
    # Read encrypted secret
    data = TOTP_SECRET_FILE.read_bytes()
    salt = data[:16]
    iv = data[16:32]
    encrypted_secret = data[32:]
    
    # Derive key from password
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(password.encode())
    
    # Decrypt
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    try:
        decrypted = decryptor.update(encrypted_secret) + decryptor.finalize()
        # Remove padding
        pad_length = decrypted[-1]
        totp_secret = decrypted[:-pad_length].decode()
        return totp_secret
    except Exception:
        raise ValueError("Invalid password")


def verify_2fa_code(totp_secret: str, code: str) -> bool:
    """Verify TOTP code."""
    totp = pyotp.TOTP(totp_secret)
    return totp.verify(code, valid_window=1)


def encrypt_env():
    """Encrypt .env file with password + 2FA."""
    print("\n" + "="*70)
    print("🔒 ENCRYPT .env FILE (Password + 2FA Required)")
    print("="*70 + "\n")
    
    if not ENV_FILE.exists():
        print(f"❌ Error: {ENV_FILE} not found")
        return
    
    # Get password
    password = getpass.getpass("🔑 Enter master password: ")
    
    # Load TOTP secret
    try:
        totp_secret = load_totp_secret(password)
    except ValueError:
        print("❌ Invalid password!")
        return
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Get 2FA code
    code = input("🔢 Enter 2FA code from your authenticator app: ").strip()
    
    if not verify_2fa_code(totp_secret, code):
        print("❌ Invalid 2FA code!")
        return
    
    print("✅ Authentication successful!\n")
    
    # Generate encryption key combining password + TOTP
    combined_secret = f"{password}:{totp_secret}:{code}"
    salt = os.urandom(16)
    
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(combined_secret.encode())
    
    # Save salt
    SALT_FILE.write_bytes(salt)
    
    # Read .env content
    content = ENV_FILE.read_bytes()
    
    # Encrypt
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Pad content
    pad_length = 16 - (len(content) % 16)
    padded_content = content + bytes([pad_length] * pad_length)
    
    encrypted = encryptor.update(padded_content) + encryptor.finalize()
    
    # Save with IV prepended
    ENCRYPTED_FILE.write_bytes(iv + encrypted)
    
    print(f"✅ Encrypted and saved to: {ENCRYPTED_FILE}")
    print(f"🔑 Salt saved to: {SALT_FILE}")
    print(f"\n💡 Safe to delete: {ENV_FILE} (keep a backup first!)")
    print(f"💡 Safe to commit: {ENCRYPTED_FILE}, {SALT_FILE}, {TOTP_SECRET_FILE}")


def decrypt_env():
    """Decrypt .env.encrypted with password + 2FA."""
    print("\n" + "="*70)
    print("🔓 DECRYPT .env FILE (Password + 2FA Required)")
    print("="*70 + "\n")
    
    if not ENCRYPTED_FILE.exists():
        print(f"❌ Error: {ENCRYPTED_FILE} not found")
        return
    
    if not SALT_FILE.exists():
        print(f"❌ Error: {SALT_FILE} not found")
        return
    
    # Get password
    password = getpass.getpass("🔑 Enter master password: ")
    
    # Load TOTP secret
    try:
        totp_secret = load_totp_secret(password)
    except ValueError:
        print("❌ Invalid password!")
        return
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Get current 2FA code
    code = input("🔢 Enter current 2FA code from your app: ").strip()
    
    if not verify_2fa_code(totp_secret, code):
        print("❌ Invalid 2FA code!")
        return
    
    print("✅ Authentication successful!\n")
    
    # Read salt
    salt = SALT_FILE.read_bytes()
    
    # Derive key (same as encryption)
    combined_secret = f"{password}:{totp_secret}:{code}"
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(combined_secret.encode())
    
    # Read encrypted data
    encrypted_data = ENCRYPTED_FILE.read_bytes()
    iv = encrypted_data[:16]
    encrypted = encrypted_data[16:]
    
    # Decrypt
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    try:
        decrypted = decryptor.update(encrypted) + decryptor.finalize()
        
        # Remove padding
        pad_length = decrypted[-1]
        content = decrypted[:-pad_length]
        
        # Save
        ENV_FILE.write_bytes(content)
        
        print(f"✅ Decrypted to: {ENV_FILE}")
        print(f"🚀 Ready to run: streamlit run app.py\n")
        
    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        print("⚠️  Check your password and 2FA code")


def verify_2fa():
    """Test 2FA without encryption/decryption."""
    print("\n" + "="*70)
    print("🔍 VERIFY 2FA CODE")
    print("="*70 + "\n")
    
    password = getpass.getpass("🔑 Enter master password: ")
    
    try:
        totp_secret = load_totp_secret(password)
    except ValueError:
        print("❌ Invalid password!")
        return
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    totp = pyotp.TOTP(totp_secret)
    current_code = totp.now()
    time_remaining = 30 - (int(time.time()) % 30)
    
    print(f"✅ Password verified!")
    print(f"\n📱 Current valid code: {current_code}")
    print(f"⏱️  Time remaining: {time_remaining} seconds\n")
    
    code = input("🔢 Enter code to verify: ").strip()
    
    if verify_2fa_code(totp_secret, code):
        print("✅ Code is valid!")
    else:
        print("❌ Code is invalid!")


def main():
    commands = {
        "setup": setup_2fa,
        "encrypt": encrypt_env,
        "decrypt": decrypt_env,
        "verify": verify_2fa,
    }
    
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage:")
        print("  python encrypt_env_2fa.py setup     # Initial 2FA setup")
        print("  python encrypt_env_2fa.py encrypt   # Encrypt .env")
        print("  python encrypt_env_2fa.py decrypt   # Decrypt .env")
        print("  python encrypt_env_2fa.py verify    # Test 2FA code")
        sys.exit(1)
    
    command = sys.argv[1]
    commands[command]()


if __name__ == "__main__":
    main()

