# src/dashboard/auth.py
"""Password hashing and verification for dashboard authentication."""

import bcrypt


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt. Use this to generate ADMIN_PASSWORD_HASH."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.dashboard.auth <password>")
        print("Outputs: bcrypt hash to set as ADMIN_PASSWORD_HASH env var")
        sys.exit(1)
    print(hash_password(sys.argv[1]))
