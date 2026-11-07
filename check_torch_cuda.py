import sys
import subprocess

def check_gpu_availability():
    """
    Check if GPU is available and properly configured
    """
    try:
        import torch
        
        print("🔍 GPU DIAGNOSTICS")
        print("=" * 40)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print()
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_properties = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_properties.total_memory / 1024**3
                compute_capability = f"{gpu_properties.major}.{gpu_properties.minor}"
                
                print(f"GPU {i}: {gpu_name}")
                print(f"   • Memory: {gpu_memory:.1f} GB")
                print(f"   • Compute capability: {compute_capability}")
                
            print()
            
            # Test GPU functionality
            try:
                device = torch.device('cuda')
                test_tensor = torch.randn(1000, 1000).to(device)
                result = torch.mm(test_tensor, test_tensor)
                print("✅ GPU test: Success - GPU is working properly!")
                return True
            except Exception as e:
                print(f"❌ GPU test failed: {e}")
                return False
        else:
            print("❌ No CUDA-capable GPU detected")
            
            # Check if PyTorch was installed with CPU-only
            if hasattr(torch.version, 'cuda') and torch.version.cuda is None:
                print("💡 PyTorch appears to be CPU-only version")
            
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def install_gpu_pytorch():
    """
    Install GPU-enabled PyTorch if needed
    """
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ GPU-enabled PyTorch already installed")
            return True
    except ImportError:
        print("📦 PyTorch not found")
    
    print("\n📦 PYTORCH GPU INSTALLATION")
    print("=" * 40)
    print("Installing GPU-enabled PyTorch...")
    print("🔗 Source: https://pytorch.org/get-started/locally/")
    print()
    print("💡 Command to run:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    
    install_now = input("Install GPU PyTorch automatically? (y/n): ").lower().strip()
    
    if install_now == 'y':
        try:
            print("🔄 Installing PyTorch with CUDA support...")
            print("   This may take a few minutes...")
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                print("✅ GPU PyTorch installation completed!")
                print("\n🔄 Restarting to verify installation...")
                
                # Re-check GPU availability
                if check_gpu_availability():
                    print("🎉 GPU PyTorch is now working!")
                    return True
                else:
                    print("⚠️ Installation completed but GPU still not detected")
                    print("   Your system might not have a CUDA-compatible GPU")
                    return False
            else:
                print(f"❌ Installation failed:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Installation timed out. Please try manually.")
            return False
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
    else:
        print("⏭️ Installation skipped")
        print("\n💡 To install manually, run:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False

def check_system_gpu():
    """
    Check if system has any GPU hardware
    """
    print("\n🖥️ SYSTEM GPU CHECK")
    print("=" * 40)
    
    try:
        # Try to detect NVIDIA GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected:")
            # Extract GPU info from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                    print(f"   • {line.strip()}")
            return True
        else:
            print("❌ nvidia-smi not found - no NVIDIA GPU or drivers missing")
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found - no NVIDIA GPU or drivers missing")
    except Exception as e:
        print(f"⚠️ Could not check for NVIDIA GPU: {e}")
    
    # Check for integrated graphics (basic check)
    try:
        import platform
        if platform.system() == "Windows":
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout
                if 'Intel' in gpu_info or 'AMD' in gpu_info:
                    print("💡 Integrated graphics detected (limited CUDA support)")
                    return False
        
    except Exception:
        pass
    
    print("❌ No suitable GPU hardware detected")
    return False

def main():
    """
    Main diagnostic and installation routine
    """
    print("🎯 PYTORCH CUDA DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Step 1: Check current GPU setup
    gpu_working = check_gpu_availability()
    
    if gpu_working:
        print("\n🎉 Your GPU setup is working perfectly!")
        print("   Ready for fast YOLO training! 🚀")
        return
    
    # Step 2: Check if system has GPU hardware
    has_gpu_hardware = check_system_gpu()
    
    if not has_gpu_hardware:
        print("\n⚠️ RECOMMENDATION:")
        print("   Your system appears to lack CUDA-compatible GPU hardware.")
        print("   Training will use CPU (much slower).")
        print("   Consider using Google Colab or cloud GPU for faster training.")
        return
    
    # Step 3: Offer to install GPU PyTorch
    print("\n💡 RECOMMENDATION:")
    print("   You have GPU hardware but PyTorch can't access it.")
    print("   Installing GPU-enabled PyTorch should fix this.")
    
    proceed = input("\nProceed with GPU PyTorch installation? (y/n): ").lower().strip()
    
    if proceed == 'y':
        success = install_gpu_pytorch()
        if success:
            print("\n🎉 Setup complete! Your system is ready for GPU training.")
        else:
            print("\n❌ Setup incomplete. You may need to install GPU drivers first.")
            print("   Visit: https://developer.nvidia.com/cuda-downloads")
    else:
        print("\n📝 Manual installation instructions:")
        print("   1. Install NVIDIA drivers if needed")
        print("   2. Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Re-run this script to verify")

if __name__ == "__main__":
    main()