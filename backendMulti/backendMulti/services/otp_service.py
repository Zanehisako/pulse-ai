import random
from django.core.cache import cache

OTP_EXPIRY = 300

def generate_otp():
    return str(random.randint(1000, 9999))

def store_otp(email, otp):
    cache.set(f"otp:{email}", otp, timeout=OTP_EXPIRY)

def verify_otp(email, otp):
    stored = cache.get(f"otp:{email}")
    return stored == otp

def mark_otp_verified(email):
    cache.set(f"otp_verified:{email}", True, timeout=600)

def is_otp_verified(email):
    return cache.get(f"otp_verified:{email}") is True

def clear_otp(email):
    cache.delete(f"otp:{email}")
    cache.delete(f"otp_verified:{email}")