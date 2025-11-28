# add_police_user.py
import streamlit_authenticator as stauth

def add_police_user():
    print("=== Add New Police User ===")
    username = input("Enter username (e.g., police3): ").strip()
    name = input("Enter full name (e.g., Officer Three): ").strip()
    password = input("Enter plaintext password (e.g., p3@123): ").strip()

    # Generate hashed password
    hashed_pw = stauth.Hasher([password]).generate()[0]

    # Output dictionary entry for frontend
    print("\n✅ Add the following entry to your 'credentials' dict in Crime_buster_frontend.py:\n")
    print(f'"{username}": {{"name": "{name}", "password": "{hashed_pw}", "role": "police"}}\n')

if __name__ == "__main__":
    add_police_user()
