import os

def print_tree(start_path, prefix=""):
    for item in os.listdir(start_path):
        path = os.path.join(start_path, item)
        print(prefix + "├── " + item)
        if os.path.isdir(path):
            print_tree(path, prefix + "│   ")

# Run from the root of your project
print_tree(".")
