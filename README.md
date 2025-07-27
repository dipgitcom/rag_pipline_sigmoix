# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Install Java (required for tabula-py)
# Ubuntu/Debian: sudo apt-get install default-jre
# macOS: brew install openjdk
# Windows: Download from Oracle or OpenJDK
