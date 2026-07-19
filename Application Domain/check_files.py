import os

current_dir = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(current_dir, "static")

print(f"Checking folder: {static_path}")
if os.path.exists(static_path):
    print("Files found inside:")
    for file in os.listdir(static_path):
        print(f" - {file}")
else:
    print("❌ The 'static' folder does not even exist here!")