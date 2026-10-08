from pathlib import Path
import base64
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


ROOT = Path(__file__).resolve().parent.parent
PRIVATE_DIR = ROOT / ".aria-private"


def main() -> None:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    private_path = PRIVATE_DIR / "jwt-private.pem"
    public_path = PRIVATE_DIR / "jwt-public.pem"
    if private_path.exists() or public_path.exists():
        raise SystemExit("Refusing to overwrite existing JWT keys")
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    print(f"Created {private_path.relative_to(ROOT)} and {public_path.relative_to(ROOT)}")
    print(f"ARIA_OAUTH_ENCRYPTION_KEY={Fernet.generate_key().decode('ascii')}")


if __name__ == "__main__":
    main()
