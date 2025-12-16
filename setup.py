import os
import sys
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create all necessary directories."""
    directories = [
        'data',
        'analytics',
        'alerts',
        'utils',
        'data_store',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory not in ['data_store', 'logs']:
            init_file = Path(directory) / '__init__.py'
            if not init_file.exists():
                init_file.write_text('"""Package initialization."""\n')
    
    print("✅ Directory structure created")

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Please run manually: pip install -r requirements.txt")

def create_init_files():
    """Create __init__.py files for each package."""
    packages = {
        'data': 'Data ingestion and storage modules',
        'analytics': 'Analytics and statistical computation modules',
        'alerts': 'Alert engine and monitoring',
        'utils': 'Utility functions and helpers'
    }
    
    for package, description in packages.items():
        init_file = Path(package) / '__init__.py'
        if not init_file.exists():
            content = f'"""{description}"""\n'
            init_file.write_text(content)
    
    print("✅ Package __init__.py files created")

def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Data and logs
data_store/
logs/
*.db
*.db-journal

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
*.csv
"""
    
    Path('.gitignore').write_text(gitignore_content.strip())
    print("✅ .gitignore created")

def verify_installation():
    """Verify that key dependencies are importable."""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'websocket',
        'sqlalchemy',
        'statsmodels',
        'sklearn',
        'scipy'
    ]
    
    failed = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        print("Please install manually: pip install " + " ".join(failed))
        return False
    else:
        print("\n✅ All dependencies verified")
        return True

def print_next_steps():
    """Print instructions for running the application."""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. Review and customize config.py")
    print("   - Adjust symbols, timeframes, windows")
    print("\n2. Start the application:")
    print("   python -m streamlit run app.py")
    print("\n3. Open your browser:")
    print("   http://localhost:8501")
    print("\n4. In the sidebar:")
    print("   - Enter trading symbols (e.g., btcusdt,ethusdt)")
    print("   - Click 'Start' to begin data collection")
    print("   - Wait 30-60 seconds for initial data")
    print("\n5. Explore the analytics tabs!")
    print("\n💡 Tips:")
    print("   - Use at least 2 symbols for pairs trading analysis")
    print("   - Minimum 100 data points for cointegration tests")
    print("   - Set up custom alerts in the Alerts tab")
    print("\n📚 Read README.md for detailed documentation")
    print("="*60 + "\n")

def main():
    """Main setup function."""
    print("\n⚡ QuanFusion Trading Analytics - Setup")
    print("="*60 + "\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        print(f"Current version: {sys.version}")
        return
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Create structure
    create_directory_structure()
    create_init_files()
    create_gitignore()
    
    # Install dependencies
    install_choice = input("\n📦 Install dependencies now? (y/n): ").lower()
    if install_choice == 'y':
        install_dependencies()
        
        # Verify
        if verify_installation():
            print_next_steps()
        else:
            print("\n⚠️  Some dependencies failed to install.")
            print("Please fix errors and run setup.py again.")
    else:
        print("\n⚠️  Remember to install dependencies:")
        print("pip install -r requirements.txt")
        print_next_steps()

if __name__ == '__main__':
    main()