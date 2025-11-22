#!/usr/bin/env python3
"""
Unified .env Encryption Tool with Optional 2FA
Supports both simple password encryption and password + 2FA (TOTP)

Usage:
    # Simple password-based encryption
    python encrypt_env.py encrypt              # Encrypt with password only
    python encrypt_env.py decrypt              # Decrypt with password only
    
    # Enhanced 2FA encryption (password + authenticator app)
    python encrypt_env.py setup-2fa            # Initial 2FA setup (one-time)
    python encrypt_env.py encrypt --2fa        # Encrypt with password + 2FA
    python encrypt_env.py decrypt --2fa        # Decrypt with password + 2FA
    python encrypt_env.py verify-2fa           # Test your 2FA code
    python encrypt_env.py disable-2fa          # Remove 2FA (keep password)
"""

import os
import sys
import getpass
import time
import tempfile
from pathlib import Path
from typing import Tuple, Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# Optional 2FA imports (only needed for 2FA features)
try:
    import pyotp
    import qrcode
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False

# File paths
ENV_FILE = Path(".env")
ENCRYPTED_FILE = Path(".env.encrypted")
SALT_FILE = Path(".env.salt")
TOTP_SECRET_FILE = Path(".env.2fa.secret")
SETUP_METHOD_FILE = Path(".env.setup_method")
ENCRYPTION_MODE_FILE = Path(".env.encryption_mode")  # Tracks if 2FA was used


def check_2fa_available():
    """Check if 2FA dependencies are installed."""
    if not TOTP_AVAILABLE:
        print("❌ 2FA features require additional packages.")
        print("   Install them with: pip install pyotp qrcode[pil]")
        sys.exit(1)


# ============================================================================
# CORE ENCRYPTION/DECRYPTION (Works with or without 2FA)
# ============================================================================

def generate_encryption_key(password: str, salt: bytes, totp_secret: Optional[str] = None, 
                           totp_code: Optional[str] = None) -> bytes:
    """
    Generate encryption key from password, optionally combined with 2FA.
    
    Args:
        password: Master password
        salt: Random salt for key derivation
        totp_secret: Optional TOTP secret for 2FA
        totp_code: Optional current TOTP code for 2FA
    
    Returns:
        32-byte encryption key
    """
    if totp_secret and totp_code:
        # Enhanced security: combine password + TOTP secret + current code
        combined_secret = f"{password}:{totp_secret}:{totp_code}"
    else:
        # Basic security: password only
        combined_secret = password
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return kdf.derive(combined_secret.encode())


def encrypt_file(content: bytes, key: bytes) -> bytes:
    """Encrypt content with AES-256-CBC."""
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Pad content to block size
    pad_length = 16 - (len(content) % 16)
    padded_content = content + bytes([pad_length] * pad_length)
    
    encrypted = encryptor.update(padded_content) + encryptor.finalize()
    
    # Return IV + encrypted data
    return iv + encrypted


def decrypt_file(encrypted_data: bytes, key: bytes) -> bytes:
    """Decrypt content with AES-256-CBC."""
    iv = encrypted_data[:16]
    encrypted = encrypted_data[16:]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    decrypted = decryptor.update(encrypted) + decryptor.finalize()

    # Validate and remove PKCS7 padding
    if len(decrypted) == 0:
        raise ValueError("Decrypted data is empty")

    pad_length = decrypted[-1]
    if pad_length > 16 or pad_length == 0:
        raise ValueError("Invalid padding length")

    # Check that all padding bytes are correct
    padding_start = len(decrypted) - pad_length
    for i in range(padding_start, len(decrypted)):
        if decrypted[i] != pad_length:
            raise ValueError("Invalid padding")

    return decrypted[:-pad_length]


# ============================================================================
# 2FA MANAGEMENT (TOTP)
# ============================================================================

def generate_totp_secret_encrypted(password: str, project_name: str = "chat4MonteCarlo") -> Tuple[str, str]:
    """
    Generate and encrypt TOTP secret for 2FA.
    Used by both CLI and UI for consistency.
    
    Args:
        password: Master password to encrypt TOTP secret
        project_name: Name to display in authenticator app
    
    Returns:
        Tuple[str, str]: (totp_secret, totp_uri)
    """
    check_2fa_available()
    
    # Generate TOTP secret
    totp_secret = pyotp.random_base32()
    
    # Create TOTP URI for QR code
    totp_uri = pyotp.totp.TOTP(totp_secret).provisioning_uri(
        name=f"{project_name}@env",
        issuer_name=f"{project_name} Environment"
    )
    
    # Encrypt the TOTP secret with password
    salt = os.urandom(16)
    key = generate_encryption_key(password, salt)
    encrypted_secret = encrypt_file(totp_secret.encode(), key)
    
    # Save encrypted secret with salt prepended
    TOTP_SECRET_FILE.write_bytes(salt + encrypted_secret)
    
    return totp_secret, totp_uri


def load_totp_secret(password: str) -> str:
    """Load and decrypt TOTP secret."""
    if not TOTP_SECRET_FILE.exists():
        raise FileNotFoundError(
            "2FA not set up. Run: python encrypt_env.py setup-2fa"
        )
    
    # Read encrypted secret
    data = TOTP_SECRET_FILE.read_bytes()
    salt = data[:16]
    encrypted_secret = data[16:]
    
    # Derive key from password
    key = generate_encryption_key(password, salt)
    
    try:
        totp_secret = decrypt_file(encrypted_secret, key).decode()
        return totp_secret
    except Exception:
        raise ValueError("Invalid password")


def verify_2fa_code(totp_secret: str, code: str) -> bool:
    """Verify TOTP code (allows ±30 seconds window)."""
    check_2fa_available()
    totp = pyotp.TOTP(totp_secret)
    return totp.verify(code, valid_window=1)


def is_2fa_enabled() -> bool:
    """Check if 2FA is currently configured."""
    return TOTP_SECRET_FILE.exists()


def get_encryption_mode() -> str:
    """Get the mode used for last encryption (password or 2fa)."""
    if ENCRYPTION_MODE_FILE.exists():
        return ENCRYPTION_MODE_FILE.read_text().strip()
    # For backward compatibility, check if encrypted file exists
    if ENCRYPTED_FILE.exists():
        return "2fa" if is_2fa_enabled() else "password"
    return "none"


# ============================================================================
# PUBLIC API (for programmatic use by other modules)
# ============================================================================

def encrypt_env(password: str = None, skip_confirm: bool = False, use_2fa: bool = False, 
                totp_code: str = None) -> bool:
    """
    Encrypt .env file (for programmatic use).
    
    Args:
        password: Password to use (if None, will prompt)
        skip_confirm: Skip password confirmation
        use_2fa: Use 2FA encryption
        totp_code: 2FA code (if use_2fa=True)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not ENV_FILE.exists():
        print(f"❌ Error: {ENV_FILE} not found")
        return False
    
    # Get password
    if password is None:
        password = getpass.getpass("🔑 Enter master password: ")
        if not skip_confirm:
            confirm = getpass.getpass("🔑 Confirm password: ")
        else:
            confirm = password
    else:
        confirm = password if skip_confirm else password
    
    if password != confirm:
        print("❌ Passwords don't match!")
        return False
    
    totp_secret = None
    
    # Handle 2FA if requested
    if use_2fa:
        if not is_2fa_enabled():
            print("❌ 2FA not set up.")
            return False
        
        try:
            totp_secret = load_totp_secret(password)
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Error: {e}")
            return False
        
        if totp_code is None:
            totp_code = input("🔢 Enter 2FA code: ").strip()
        
        if not verify_2fa_code(totp_secret, totp_code):
            print("❌ Invalid 2FA code!")
            return False
    
    try:
        # Generate encryption key
        salt = os.urandom(16)
        key = generate_encryption_key(password, salt, totp_secret, totp_code)
        
        # Read and encrypt .env content
        content = ENV_FILE.read_bytes()
        encrypted_data = encrypt_file(content, key)
        
        # Save encrypted file and salt
        ENCRYPTED_FILE.write_bytes(encrypted_data)
        SALT_FILE.write_bytes(salt)
        
        # Record encryption mode
        ENCRYPTION_MODE_FILE.write_text("2fa" if use_2fa else "password")
        
        # Delete original .env for security
        ENV_FILE.unlink()
        
        return True
    except Exception as e:
        print(f"❌ Encryption failed: {e}")
        return False


def decrypt_env(password: str = None, use_2fa: bool = None, totp_code: str = None) -> bool:
    """
    Decrypt .env.encrypted (for programmatic use).
    
    Args:
        password: Password to use (if None, will prompt)
        use_2fa: Use 2FA (if None, auto-detect)
        totp_code: 2FA code (if use_2fa=True)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not ENCRYPTED_FILE.exists() or not SALT_FILE.exists():
        print("❌ Error: Encrypted files not found")
        return False

    # Check for and clean up any corrupted .env file from previous interrupted decryption
    if ENV_FILE.exists():
        try:
            content = ENV_FILE.read_text(encoding='utf-8')
            # Basic validation - check if it looks like a valid .env file
            if not content.strip() or not any(line.strip() and not line.strip().startswith('#') for line in content.split('\n')):
                print("⚠️  Found potentially corrupted .env file, removing...")
                ENV_FILE.unlink()
        except (UnicodeDecodeError, OSError) as e:
            print(f"⚠️  Found corrupted .env file ({e}), removing...")
            ENV_FILE.unlink()

    # Auto-detect encryption mode if not specified
    if use_2fa is None:
        encryption_mode = get_encryption_mode()
        use_2fa = (encryption_mode == "2fa")
    
    # Get password
    if password is None:
        password = getpass.getpass("🔑 Enter master password: ")
    
    totp_secret = None
    
    # Handle 2FA if needed
    if use_2fa:
        try:
            totp_secret = load_totp_secret(password)
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Error: {e}")
            return False
        
        if totp_code is None:
            totp_code = input("🔢 Enter 2FA code: ").strip()
        
        if not verify_2fa_code(totp_secret, totp_code):
            print("❌ Invalid 2FA code!")
            return False
    
    try:
        # Read salt and encrypted data
        salt = SALT_FILE.read_bytes()
        encrypted_data = ENCRYPTED_FILE.read_bytes()

        # Generate key and decrypt
        key = generate_encryption_key(password, salt, totp_secret, totp_code)
        content = decrypt_file(encrypted_data, key)

        # Basic validation of decrypted content
        try:
            content_str = content.decode('utf-8')
            # Check if it looks like a .env file (has key=value pairs)
            lines = [line.strip() for line in content_str.split('\n') if line.strip() and not line.strip().startswith('#')]
            if not lines:
                raise ValueError("Decrypted content appears to be empty")
            valid_lines = [line for line in lines if '=' in line]
            if len(valid_lines) == 0:
                raise ValueError("Decrypted content doesn't contain valid key=value pairs")
        except UnicodeDecodeError:
            raise ValueError("Decrypted content is not valid UTF-8 text")

        # Atomic write: write to temp file first, then move
        with tempfile.NamedTemporaryFile(mode='wb', dir=ENV_FILE.parent, delete=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Force write to disk
            temp_path = Path(temp_file.name)

        # Atomically replace the target file
        temp_path.replace(ENV_FILE)
        return True

    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        return False


def setup_2fa(password: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Setup 2FA programmatically (for use by UI).
    
    Args:
        password: Master password (if None, will prompt)
    
    Returns:
        Tuple[Optional[str], Optional[str]]: (totp_secret, totp_uri) or (None, None) on failure
    """
    check_2fa_available()
    
    if password is None:
        password = getpass.getpass("🔑 Set master password: ")
        confirm = getpass.getpass("🔑 Confirm password: ")
        if password != confirm:
            print("❌ Passwords don't match!")
            return None, None
    
    try:
        project_name = "chat4MonteCarlo"
        return generate_totp_secret_encrypted(password, project_name)
    except Exception as e:
        print(f"❌ 2FA setup failed: {e}")
        return None, None


# ============================================================================
# CLI COMMANDS
# ============================================================================

def cmd_setup_2fa():
    """Setup 2FA: Generate TOTP secret and show QR code."""
    check_2fa_available()
    
    print("\n" + "="*70)
    print("🔐 2FA SETUP - Two-Factor Authentication")
    print("="*70 + "\n")
    
    # Check if already configured
    if TOTP_SECRET_FILE.exists():
        setup_method = SETUP_METHOD_FILE.read_text().strip() if SETUP_METHOD_FILE.exists() else "unknown"
        
        print("⚠️  WARNING: 2FA is already configured!")
        print(f"   Setup method: {setup_method}")
        
        if setup_method == "webui":
            print("\n   ❌ This was set up via the Web UI (Streamlit).")
            print("   You should use the Web UI to reset:")
            print("   → Run: streamlit run app.py")
            print("   → Click 'Reset Credentials' in the sidebar\n")
            return
        
        print("   Resetting will invalidate your current authenticator app setup.")
        print("   You will need to scan a NEW QR code.\n")
        response = input("   Are you sure you want to reset? (type 'YES' to confirm): ")
        if response != "YES":
            print("✅ Setup cancelled. Existing 2FA configuration preserved.")
            return
    
    # Get password
    password = getpass.getpass("\n🔑 Set a master password to protect 2FA: ")
    confirm = getpass.getpass("🔑 Confirm password: ")
    
    if password != confirm:
        print("❌ Passwords don't match!")
        return
    
    # Generate and encrypt TOTP secret
    project_name = "chat4MonteCarlo"
    totp_secret, totp_uri = generate_totp_secret_encrypted(password, project_name)
    
    # Generate QR code for terminal display
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    
    print("\n📱 Scan this QR code with your authenticator app:")
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
    
    # Record setup method
    SETUP_METHOD_FILE.write_text("terminal")
    
    print(f"\n✅ 2FA setup complete!")
    print(f"📁 Secret saved to: {TOTP_SECRET_FILE}")
    print(f"\nNext steps:")
    print("  python encrypt_env.py encrypt --2fa    # Encrypt with 2FA")
    print("  python encrypt_env.py encrypt           # Or use password only")


def cmd_verify_2fa():
    """Test 2FA code without encryption/decryption."""
    check_2fa_available()
    
    print("\n" + "="*70)
    print("🔍 VERIFY 2FA CODE")
    print("="*70 + "\n")
    
    if not is_2fa_enabled():
        print("❌ 2FA not set up. Run: python encrypt_env.py setup-2fa")
        return
    
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


def cmd_disable_2fa():
    """Disable 2FA (removes 2FA files, keeps password encryption)."""
    print("\n" + "="*70)
    print("🔓 DISABLE 2FA")
    print("="*70 + "\n")
    
    if not is_2fa_enabled():
        print("ℹ️  2FA is not currently enabled.")
        return
    
    print("⚠️  WARNING: This will remove 2FA protection.")
    print("   Your .env file will still be encrypted with password only.")
    print("   You'll need to re-encrypt it without --2fa flag.\n")
    
    response = input("   Are you sure? (type 'YES' to confirm): ")
    if response != "YES":
        print("✅ Cancelled. 2FA remains enabled.")
        return
    
    # Remove 2FA files
    TOTP_SECRET_FILE.unlink(missing_ok=True)
    SETUP_METHOD_FILE.unlink(missing_ok=True)
    
    print("✅ 2FA disabled successfully!")
    print("\nTo continue using encryption:")
    print("  1. python encrypt_env.py decrypt      # Decrypt with current password")
    print("  2. python encrypt_env.py encrypt      # Re-encrypt without --2fa")


def cmd_encrypt(use_2fa: bool = False):
    """Encrypt .env file with password, optionally with 2FA."""
    print("\n" + "="*70)
    if use_2fa:
        print("🔒 ENCRYPT .env FILE (Password + 2FA)")
    else:
        print("🔒 ENCRYPT .env FILE (Password Only)")
    print("="*70 + "\n")
    
    if not ENV_FILE.exists():
        print(f"❌ Error: {ENV_FILE} not found")
        return
    
    # Get password
    password = getpass.getpass("🔑 Enter master password: ")
    confirm = getpass.getpass("🔑 Confirm password: ")
    
    if password != confirm:
        print("❌ Passwords don't match!")
        return
    
    totp_secret = None
    totp_code = None
    
    # Handle 2FA if requested
    if use_2fa:
        check_2fa_available()
        
        if not is_2fa_enabled():
            print("❌ 2FA not set up. Run: python encrypt_env.py setup-2fa")
            return
        
        try:
            totp_secret = load_totp_secret(password)
        except ValueError:
            print("❌ Invalid password!")
            return
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return
        
        totp_code = input("🔢 Enter 2FA code from your authenticator app: ").strip()
        
        if not verify_2fa_code(totp_secret, totp_code):
            print("❌ Invalid 2FA code!")
            return
        
        print("✅ 2FA authentication successful!\n")
    
    # Generate encryption key
    salt = os.urandom(16)
    key = generate_encryption_key(password, salt, totp_secret, totp_code)
    
    # Read and encrypt .env content
    content = ENV_FILE.read_bytes()
    encrypted_data = encrypt_file(content, key)
    
    # Save encrypted file and salt
    ENCRYPTED_FILE.write_bytes(encrypted_data)
    SALT_FILE.write_bytes(salt)
    
    # Record encryption mode
    ENCRYPTION_MODE_FILE.write_text("2fa" if use_2fa else "password")
    
    # Delete original .env for security
    ENV_FILE.unlink()
    
    print(f"✅ Encrypted and saved to: {ENCRYPTED_FILE}")
    print(f"🔑 Salt saved to: {SALT_FILE}")
    print(f"🗑️  Original .env file deleted for security")
    
    if use_2fa:
        print(f"\n💡 Safe to commit: {ENCRYPTED_FILE}, {SALT_FILE}, {TOTP_SECRET_FILE}")
    else:
        print(f"\n💡 Safe to commit: {ENCRYPTED_FILE}, {SALT_FILE}")


def cmd_decrypt(use_2fa: bool = None):
    """
    Decrypt .env.encrypted with password, optionally with 2FA.
    If use_2fa is None, auto-detect from encryption mode.
    """
    if not ENCRYPTED_FILE.exists():
        print(f"❌ Error: {ENCRYPTED_FILE} not found")
        return
    
    if not SALT_FILE.exists():
        print(f"❌ Error: {SALT_FILE} not found")
        return
    
    # Auto-detect encryption mode if not specified
    if use_2fa is None:
        encryption_mode = get_encryption_mode()
        use_2fa = (encryption_mode == "2fa")
    
    print("\n" + "="*70)
    if use_2fa:
        print("🔓 DECRYPT .env FILE (Password + 2FA)")
    else:
        print("🔓 DECRYPT .env FILE (Password Only)")
    print("="*70 + "\n")
    
    # Get password
    password = getpass.getpass("🔑 Enter master password: ")
    
    totp_secret = None
    totp_code = None
    
    # Handle 2FA if needed
    if use_2fa:
        check_2fa_available()
        
        try:
            totp_secret = load_totp_secret(password)
        except ValueError:
            print("❌ Invalid password!")
            return
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return
        
        totp_code = input("🔢 Enter current 2FA code: ").strip()
        
        if not verify_2fa_code(totp_secret, totp_code):
            print("❌ Invalid 2FA code!")
            return
        
        print("✅ 2FA authentication successful!\n")
    
    # Read salt and encrypted data
    salt = SALT_FILE.read_bytes()
    encrypted_data = ENCRYPTED_FILE.read_bytes()
    
    # Generate key and decrypt
    key = generate_encryption_key(password, salt, totp_secret, totp_code)
    
    try:
        content = decrypt_file(encrypted_data, key)
        ENV_FILE.write_bytes(content)
        
        print(f"✅ Decrypted to: {ENV_FILE}")
        print(f"🚀 Ready to run: streamlit run app.py\n")
        
    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        if use_2fa:
            print("⚠️  Check your password and 2FA code")
        else:
            print("⚠️  Check your password, or try: python encrypt_env.py decrypt --2fa")


# ============================================================================
# MAIN CLI INTERFACE
# ============================================================================

def print_help():
    """Print usage information."""
    print("\n" + "="*70)
    print("🔐 Unified .env Encryption Tool")
    print("="*70)
    print("\n📌 BASIC USAGE (Password Only):")
    print("  python encrypt_env.py encrypt              # Encrypt with password")
    print("  python encrypt_env.py decrypt              # Decrypt with password")
    
    if TOTP_AVAILABLE:
        print("\n🔐 ENHANCED SECURITY (Password + 2FA):")
        print("  python encrypt_env.py setup-2fa            # One-time 2FA setup")
        print("  python encrypt_env.py encrypt --2fa        # Encrypt with 2FA")
        print("  python encrypt_env.py decrypt --2fa        # Decrypt with 2FA")
        print("  python encrypt_env.py verify-2fa           # Test 2FA code")
        print("  python encrypt_env.py disable-2fa          # Remove 2FA")
    else:
        print("\n💡 For 2FA support, install: pip install pyotp qrcode[pil]")
    
    print("\n💡 RECOMMENDED: Use the web UI for easier management:")
    print("   streamlit run app.py")
    print()


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    use_2fa_flag = "--2fa" in sys.argv
    
    commands = {
        "encrypt": lambda: cmd_encrypt(use_2fa=use_2fa_flag),
        "decrypt": lambda: cmd_decrypt(use_2fa=use_2fa_flag if use_2fa_flag else None),
        "setup-2fa": cmd_setup_2fa,
        "verify-2fa": cmd_verify_2fa,
        "disable-2fa": cmd_disable_2fa,
        "help": print_help,
        "--help": print_help,
        "-h": print_help,
    }
    
    if command not in commands:
        print(f"❌ Unknown command: {command}")
        print_help()
        sys.exit(1)
    
    commands[command]()


if __name__ == "__main__":
    main()
