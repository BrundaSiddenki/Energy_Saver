import json
import os
import hashlib
import streamlit as st

USER_DATA_FILE = "user_data.json"

def load_user_data():
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

def save_user_data(user_data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(user_data, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    user_data = load_user_data()
    if username in user_data:
        return False, "Username already exists."
    user_data[username] = {
        "password": hash_password(password),
        "profile": {
            "name": "",
            "email": "",
            "preferences": {}
        }
    }
    save_user_data(user_data)
    return True, "User registered successfully."

def login_user(username, password):
    user_data = load_user_data()
    if username not in user_data:
        return False, "Username does not exist."
    # Handle old format where user_data[username] is a string (hashed password)
    if isinstance(user_data[username], str):
        if user_data[username] != hash_password(password):
            return False, "Incorrect password."
        else:
            # Migrate to new format with profile
            user_data[username] = {
                "password": user_data[username],
                "profile": {
                    "name": "",
                    "email": "",
                    "preferences": {}
                }
            }
            save_user_data(user_data)
            st.session_state['username'] = username
            return True, "Login successful."
    else:
        if user_data[username]["password"] != hash_password(password):
            return False, "Incorrect password."
        st.session_state['username'] = username
        return True, "Login successful."

def logout_user():
    if 'username' in st.session_state:
        del st.session_state['username']

def is_logged_in():
    return 'username' in st.session_state

def get_user_profile(username):
    user_data = load_user_data()
    if username in user_data:
        return user_data[username].get("profile", {})
    return {}
