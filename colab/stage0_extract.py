"""
Stage 0: Multi-part archive extraction for GenImage dataset.
Extracts full sdv4 subset and validation splits of remaining 7 subsets.
"""
import os
import subprocess
import argparse

SUBSETS = ["sdv4", "sdv5", "wukong", "midjourney", "glide", "adm", "vqdm", "biggan"]

def extract_archives(raw_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for subset in SUBSETS:
        print(f"[*] Processing {subset}...")
        sub_out = os.path.join(output_dir, subset)
        os.makedirs(sub_out, exist_ok=True)
        
        # 查找分卷压缩包首包 (如 .zip.001 或 .7z.001)
        part1 = os.path.join(raw_dir, subset, f"{subset}.zip.001")
        if not os.path.exists(part1):
            part1 = os.path.join(raw_dir, f"{subset}.zip.001")
        
        if not os.path.exists(part1):
            print(f"[!] Archive {part1} not found, skipping.")
            continue
            
        cmd = ["7z", "x", part1, f"-o{sub_out}", "-y"]
        if subset != "sdv4":
            # 仅提取验证集，节约磁盘与显存空间
            cmd.append('-i!*/val/*')
            
        subprocess.run(cmd, check=True)
    print("[+] All extractions completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="/content/drive/MyDrive/GenImage_Raw")
    parser.add_argument("--output-dir", default="/content/raw_extracted")
    args = parser.parse_args()
    extract_archives(args.raw_dir, args.output_dir)