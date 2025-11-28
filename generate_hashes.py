# generate_hashes.py
import streamlit_authenticator as stauth

# Put plaintext passwords here
plaintext_passwords = [
     "p1@123",   # police1
    "p2@123"      # police2

]

# Create hasher
hasher = stauth.Hasher()

# Loop through and hash each password
for p in plaintext_passwords:
    hashed = hasher.hash(p)   # pass one password at a time
    print(f"PLAIN: {p}\nHASH : {hashed}\n")
