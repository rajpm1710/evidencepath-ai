"""Fetch original public research PDFs; no paid API or credentials required."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [{'filename': '01_attention_is_all_you_need.pdf', 'url': 'https://arxiv.org/pdf/1706.03762', 'sha256': 'bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697'}, {'filename': '02_retrieval_augmented_generation.pdf', 'url': 'https://arxiv.org/pdf/2005.11401', 'sha256': '23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233'}, {'filename': '03_bert.pdf', 'url': 'https://arxiv.org/pdf/1810.04805', 'sha256': '5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e'}, {'filename': '04_sentence_bert.pdf', 'url': 'https://arxiv.org/pdf/1908.10084', 'sha256': '92366d22fc0521350d6cbe0ad72bcee754f90e88181d7b4a8e2adb1e7e594fec'}, {'filename': '05_billion_scale_similarity_search.pdf', 'url': 'https://arxiv.org/pdf/1702.08734', 'sha256': '89b2371b261846059d416bdf1423b15a6e3cc7d1b2d5dc2729ee5f6c3399049a'}, {'filename': '06_chain_of_thought.pdf', 'url': 'https://arxiv.org/pdf/2201.11903', 'sha256': '7d9f878c23b460e4566aa4ec9201b1abfb3b8faefb2b1356e411cb90fef72a12'}, {'filename': '07_tree_of_thoughts.pdf', 'url': 'https://arxiv.org/pdf/2305.10601', 'sha256': '79c5237e3f63953a73f2b0d6894327702ee1f7e981450c251bb1b5cb4f8d7b8f'}, {'filename': '08_react_reasoning_and_acting.pdf', 'url': 'https://arxiv.org/pdf/2210.03629', 'sha256': 'f285b0971ae4a790e402fb93966bed3adde2cf0a04977d08b2b40d6ab0cace69'}]

def main():
    folder = ROOT / "documents"
    folder.mkdir(exist_ok=True)
    for source in SOURCES:
        target = folder / source["filename"]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == source["sha256"]:
            print("Already verified:", target.name)
            continue
        request = urllib.request.Request(source["url"], headers={"User-Agent": "EvidencePath-AI/1.0 (academic document retrieval)"})
        with urllib.request.urlopen(request, timeout=90) as response:
            data = response.read()
        if not data.startswith(b"%PDF"):
            raise RuntimeError("The source did not return a PDF: " + target.name)
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise RuntimeError("Source version changed. Review and update the provenance hash before using: " + target.name)
        target.write_bytes(data)
        print("Downloaded and verified:", target.name)
    print("Eight original PDFs ready. Run: python evidencepath.py")

if __name__ == "__main__":
    main()
