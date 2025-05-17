import os
import sys
import traceback

def main():
    print("Starting permission test...")
    try:
        # Test file writing
        print("Testing file write access...")
        with open("permission_test.txt", "w") as f:
            f.write("Testing write permissions\n")
        print("File write successful")
        
        # Test directory creation
        print("Testing directory creation...")
        os.makedirs("test_permission_dir", exist_ok=True)
        print("Directory creation successful")
        
        # Test file writing in new directory
        print("Testing file write in new directory...")
        with open("test_permission_dir/test.txt", "w") as f:
            f.write("Testing write permissions in directory\n")
        print("File write in directory successful")
        
        print("All permission tests passed!")
        return True
    except Exception as e:
        print(f"Error during permission test: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print("Test completed with success =", success)
