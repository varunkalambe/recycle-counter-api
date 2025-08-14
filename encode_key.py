import base64

# The name of your key file
file_name = "serviceAccountKey.json"

try:
    # Open the file in binary read mode
    with open(file_name, "rb") as key_file:
        # Read the file's content
        key_content_bytes = key_file.read()
        
        # Encode the content into Base64
        base64_encoded_key = base64.b64encode(key_content_bytes)
        
        # Print the result as a string
        print("✅ Your Base64 key is:\n")
        print(base64_encoded_key.decode('utf-8'))

except FileNotFoundError:
    print(f"🔥 Error: Make sure '{file_name}' is in the same folder as this script.")