import sys
import os

print("Simplest test script")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

with open("simple_test_output.txt", "w") as f:
    f.write("This is a simple test\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Current directory: {os.getcwd()}\n")

print("File written successfully")
