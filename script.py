import tensorflow as tf
import sys
import os # For checking environment variables

print(f"TensorFlow Version: {tf.__version__}")
print(f"Python Version: {sys.version}")

# Optional: Check relevant environment variables (example for CUDA)
cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
print(f"CUDA_PATH (or CUDA_HOME) environment variable: {cuda_path}")
# You can add more checks for PATH and LD_LIBRARY_PATH if on Linux/macOS

print("\nChecking available physical devices:")
physical_devices = tf.config.list_physical_devices()
print(f"All physical devices: {physical_devices}")

print("\nChecking available logical devices:") # Might give more insight
logical_devices = tf.config.list_logical_devices()
print(f"All logical devices: {logical_devices}")


print("\nChecking specifically for GPUs:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            # Print details about the GPU
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Details: {details.get('name', 'N/A')}, Compute Capability: {details.get('compute_capability', 'N/A')}")
        except Exception as e:
            print(f"    Could not get details for GPU {i}: {e}")

    # Set memory growth to avoid consuming all GPU memory
    try:
        for gpu_instance in gpus: # Use a different variable name to avoid confusion
            tf.config.experimental.set_memory_growth(gpu_instance, True)
        print("Memory growth set to True for all detected GPUs.")
    except RuntimeError as e:
        # This error often happens if memory growth is set after context initialization
        print(f"Error setting memory growth: {e}. This might be normal if already set or if using a different strategy.")
    except Exception as e:
        print(f"An unexpected error occurred while setting memory growth: {e}")

else:
    print("No GPU found by TensorFlow.")
    print("Please check the following:")
    print("1. NVIDIA GPU drivers are installed correctly.")
    print("2. CUDA Toolkit is installed and compatible with your TensorFlow version.")
    print("3. cuDNN library is installed and compatible with your CUDA version.")
    print("4. Environment variables (PATH, LD_LIBRARY_PATH, CUDA_PATH) are set correctly.")
    print("5. You have the 'tensorflow' package installed (not 'tensorflow-cpu' if you want GPU).")
    print("6. Your GPU has a compute capability supported by TensorFlow.")
    print("7. Run 'nvidia-smi' in your terminal to see if the OS detects the GPU and drivers.")

# You can also try a simple TensorFlow operation on the GPU to see if it works
# if gpus:
#     try:
#         print("\nAttempting a simple GPU operation:")
#         with tf.device('/GPU:0'): # Or the specific GPU you want to test
#             a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
#             b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
#             c = tf.matmul(a, b)
#         print(f"Result of matrix multiplication on GPU: {c.numpy()}")
#     except RuntimeError as e:
#         print(f"Error during GPU operation: {e}")