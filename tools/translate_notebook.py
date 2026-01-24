#!/usr/bin/env python3
"""Translate Jupyter notebook from English to Chinese using DeepSeek API."""

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from env_key_manager import APIKeyManager
except ImportError:
    print("Please install env-key-manager: pip install env-key-manager", file=sys.stderr)
    sys.exit(1)


def translate_text(text: str, api_key: str) -> str:
    """Translate text using DeepSeek API."""
    if not text.strip():
        return text
    url = "https://api.deepseek.com/v1/chat/completions"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Translate English to Chinese. Return only translated text."},
                {"role": "user", "content": text}
            ],
            "temperature": 0.3
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def translate_notebook(notebook_path: Path, api_key: str, output_path: Path = None) -> bool:
    """Translate a Jupyter notebook."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        if output_path is None:
            output_path = notebook_path.parent / f"{notebook_path.stem}_zh.ipynb"
        
        for cell in nb.get("cells", []):
            if cell["cell_type"] == "markdown":
                source = "".join(cell.get("source", []))
                if source.strip():
                    cell["source"] = [translate_text(source, api_key) + "\n"]
            elif cell["cell_type"] == "code":
                lines = cell.get("source", [])
                translated = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("#") and len(stripped) > 1:
                        comment = stripped[1:].strip()
                        if comment:
                            translated.append(f"# {translate_text(comment, api_key)}\n")
                        else:
                            translated.append(line)
                    else:
                        translated.append(line)
                cell["source"] = translated
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        
        return True
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False


# Setup API key using env-key-manager
key_manager = APIKeyManager()
key_manager.setup_api_key(["DEEPSEEK_API_KEY"])
API_KEY = key_manager.load_custom_api_key("DEEPSEEK_API_KEY")
if not API_KEY:
    print("Error: Failed to get DeepSeek API key from env-key-manager", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Jupyter notebook to Chinese")
    parser.add_argument("notebook", type=str, help="Path to .ipynb file")
    parser.add_argument("-o", "--output", type=str, help="Output file path (default: original_name_zh.ipynb)")
    
    args = parser.parse_args()
    
    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"Error: File not found: {notebook_path}", file=sys.stderr)
        sys.exit(1)
    
    if not notebook_path.suffix == ".ipynb":
        print(f"Error: Not a Jupyter notebook file: {notebook_path}", file=sys.stderr)
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else None
    print(f"Translating: {notebook_path.name}")
    
    if translate_notebook(notebook_path, API_KEY, output_path):
        print(f"✓ Saved: {output_path or notebook_path.parent / f'{notebook_path.stem}_zh.ipynb'}")
    else:
        sys.exit(1)

