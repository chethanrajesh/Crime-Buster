import bcrypt
hash_pw = b"$2b$12$IleflOKbG3WEGHTNBmRt..DqLPNfP1B7xJ7q8kt95PewLWU0QU0X."
print(bcrypt.checkpw(b"p2@123",hash_pw)) 