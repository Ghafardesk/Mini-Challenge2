import os
import sys
from huggingface_hub import snapshot_download, HfApi

# Model target dari Hugging Face
# Catatan: Bisa disesuaikan ke 'meta-llama/Llama-3.2-11B-Vision-Instruct' atau repo kustom Anda
MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-11B-Vision-Instruct")
CACHE_DIR = os.path.join(os.getcwd(), "models")

def download_llama_model():
    print(f"[Installer] Memulai proses pengunduhan model: {MODEL_ID}")
    print(f"[Installer] Lokasi penyimpanan lokal: {CACHE_DIR}")
    
    # Ambil token Hugging Face dari Environment Variable (Aman)
    hf_token = os.getenv("HF_TOKEN")
    
    # Verifikasi kredensial
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        print(f"[Auth] Logged in sebagai akun Hugging Face: {user_info['name']}")
    except Exception as e:
        print(f"[Warning] Kredensial HF Token tidak terdeteksi atau gagal: {e}")
        print("[Info] Pastikan Anda telah menjalankan 'huggingface-cli login' atau mengatur 'export HF_TOKEN=hf_...'")

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        print("[Downloading] Mengunduh snapshot model dari Hugging Face Hub...")
        
        local_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=os.path.join(CACHE_DIR, MODEL_ID.replace("/", "_")),
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"], # Abaikan format yang tidak dibutuhkan
            resume_download=True
        )
        print(f"\n[Success] Model berhasil diunduh dan tersimpan di: {local_path}")
    except Exception as e:
        print(f"\n[Error] Gagal mengunduh model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_llama_model()
