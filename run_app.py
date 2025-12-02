import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "index.pkl")

def run_command(cmd_list):
    """Runs a subprocess command and ensures proper error reporting."""
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {' '.join(cmd_list)}")
        print(e)
        sys.exit(1)

def main():
    print("\n📚 Starting RAG Study Buddy...\n")
    # Check if embedding index exists; if not, build it
    if not os.path.exists(INDEX_PATH):
        print("🔍 No embedding index found. Building index first...\n")
        run_command([sys.executable, "-m", "app.build_index"])
    else:
        print("✅ Embedding index found.\n")

    # launch the CLI app
    print("🚀 Launching CLI assistant...\n")
    try:
        run_command([sys.executable, "-m", "app.rag_cli"])
    except KeyboardInterrupt:

        sys.exit(0)

if __name__ == "__main__":
    main()
