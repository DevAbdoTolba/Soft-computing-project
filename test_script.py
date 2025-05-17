import sys
import os

print("Hello world")
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir("."))

with open("test_output.txt", "w") as f:
    f.write("Test successful\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Current directory: {os.getcwd()}\n")
