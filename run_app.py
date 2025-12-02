import subprocess
import os

BASE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE, "embeddings", "index.pkl")

def main():
    if not os.path.exists(INDEX_PATH):
        print("Index not found — building...")
        subprocess.run(["python", "-m", "app.build_index"], check=True)

    subprocess.run(["python", "-m", "app.rag_cli"], check=True)

if __name__ == "__main__":
    main()
