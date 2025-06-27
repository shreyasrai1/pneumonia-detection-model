import requests

url = "http://127.0.0.1:5000/predict"

# ✅ This image exists based on your screenshot
image_path = r"C:\Users\shrey\Desktop\jupyter\pneumonia detection model\test\NORMAL\IM-0033-0001-0002.jpeg"

with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print(response.json())
