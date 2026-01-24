import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from pdf2zh import translate_stream
    from pdf2zh.doclayout import OnnxModel
    from pdf2zh.config import ConfigManager
except ImportError:
    print("Please install pdf2zh (pip install pdf2zh)", file=sys.stderr)
    sys.exit(1)


def is_translated_file(filepath: Path) -> bool:
    """Check if file is already translated (has -mono or -dual suffix)."""
    return filepath.stem.endswith(('-mono', '-dual'))


def has_translated_versions(filepath: Path) -> bool:
    """Check if translated versions already exist."""
    parent = filepath.parent
    base_name = filepath.stem
    return (parent / f"{base_name}-mono.pdf").exists() or (parent / f"{base_name}-dual.pdf").exists()


def find_untranslated_pdfs(directory: Path) -> list[Path]:
    """Find all PDF files that need translation."""
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                filepath = Path(root) / file
                if not is_translated_file(filepath) and not has_translated_versions(filepath):
                    pdf_files.append(filepath)
    return pdf_files


def get_required_envs(service: str) -> list[str]:
    """Get required environment variables for a service."""
    service_env_map = {
        'deepseek': ['DEEPSEEK_API_KEY'],
        'openai': ['OPENAI_API_KEY'],
        'azureopenai': ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_BASE_URL'],
        'zhipu': ['ZHIPU_API_KEY'],
        'gemini': ['GEMINI_API_KEY'],
        'groq': ['GROQ_API_KEY'],
        'grok': ['GROK_API_KEY'],
        'deepl': ['DEEPL_AUTH_KEY'],
        'azure': ['AZURE_API_KEY', 'AZURE_ENDPOINT'],
        'tencent': ['TENCENTCLOUD_SECRET_ID', 'TENCENTCLOUD_SECRET_KEY'],
        'modelscope': ['MODELSCOPE_API_KEY'],
        'silicon': ['SILICON_API_KEY'],
        'x302ai': ['X302AI_API_KEY'],
        'dify': ['DIFY_API_URL', 'DIFY_API_KEY'],
        'anythingllm': ['AnythingLLM_URL', 'AnythingLLM_APIKEY'],
        'qwenmt': ['ALI_API_KEY'],
    }
    return service_env_map.get(service.lower(), [])


def get_api_value(service: str, env_key: str) -> Optional[str]:
    """Get API value from environment variable or config file."""
    # Check environment variable first
    if env_key in os.environ:
        return os.environ[env_key]
    
    # Check config file
    config_envs = ConfigManager.get_translator_by_name(service.lower())
    if config_envs and env_key in config_envs:
        return config_envs[env_key]
    
    return None


def get_missing_api_configs(service: str) -> list[str]:
    """Get list of missing API configuration keys for a service."""
    required_envs = get_required_envs(service)
    if not required_envs:
        return []  # Service doesn't require API keys
    
    return [env for env in required_envs if not get_api_value(service, env)]


def check_api_config(service: str) -> bool:
    """Check if API configuration is available for the service."""
    missing = get_missing_api_configs(service)
    
    if missing:
        print(f"Warning: Missing API configuration for '{service}':", file=sys.stderr)
        for env in missing:
            print(f"  - {env}", file=sys.stderr)
        print(f"\nPlease set the environment variables or configure them in pdf2zh config.", file=sys.stderr)
        return False
    return True


def prompt_and_save_api_config(service: str) -> bool:
    """Prompt user to input missing API configuration and optionally save it."""
    missing = get_missing_api_configs(service)
    
    if not missing:
        return True  # All required configs are available
    
    print(f"\nMissing API configuration for '{service}':")
    config_values = {}
    
    # Get existing config from file
    existing_config = ConfigManager.get_translator_by_name(service.lower()) or {}
    
    for env in missing:
        existing_value = existing_config.get(env, '')
        prompt = f"Enter {env}"
        if existing_value:
            prompt += f" (current: {existing_value[:10]}...)" if len(existing_value) > 10 else f" (current: {existing_value})"
        prompt += ": "
        
        value = input(prompt).strip()
        if value:
            config_values[env] = value
            # Set environment variable for current session
            os.environ[env] = value
    
    if not config_values:
        print("No configuration provided. Translation may fail.")
        return False
    
    # Ask if user wants to save to config file
    if input("\nSave configuration to file? (y/n): ").lower() == 'y':
        # Merge with existing config
        final_config = {**existing_config, **config_values}
        ConfigManager.set_translator_by_name(service.lower(), final_config)
        config_path = Path.home() / ".config" / "PDFMathTranslate" / "config.json"
        print(f"✓ Configuration saved to: {config_path}")
    
    return True


def translate_pdf(filepath: Path, params: dict) -> bool:
    """Translate a single PDF file."""
    try:
        mono, dual = translate_stream(stream=filepath.read_bytes(), **params)
        parent_dir = filepath.parent
        base_name = filepath.stem
        (parent_dir / f"{base_name}-mono.pdf").write_bytes(mono)
        (parent_dir / f"{base_name}-dual.pdf").write_bytes(dual)
        return True
    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "connection" in error_msg.lower():
            print(f"  ✗ Connection error. Please check your network or API service status.", file=sys.stderr)
        elif "API" in error_msg or "api" in error_msg.lower() or "key" in error_msg.lower():
            print(f"  ✗ API error: {error_msg}", file=sys.stderr)
            print(f"    Please check your API configuration.", file=sys.stderr)
        else:
            print(f"  ✗ Failed: {error_msg}", file=sys.stderr)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch translate PDF files. Supports both single PDF file and directory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python batch_translate.py paper.pdf
  python batch_translate.py ./papers
  python batch_translate.py ./papers --service google --thread 8
        '''
    )
    parser.add_argument('path', type=str, nargs='?',
                        help='PDF file path or directory containing PDF files')
    parser.add_argument('--service', type=str, default='deepseek',
                        help='Translation service (default: deepseek)')
    parser.add_argument('--lang-in', type=str, default='en',
                        help='Source language code (default: en)')
    parser.add_argument('--lang-out', type=str, default='zh',
                        help='Target language code (default: zh)')
    parser.add_argument('--thread', type=int, default=12,
                        help='Number of threads (default: 12)')
    parser.add_argument('--check-api', action='store_true',
                        help='Check API configuration for the specified service (can be used alone)')
    
    args = parser.parse_args()
    
    if args.check_api:
        if check_api_config(args.service):
            print(f"✓ API configuration for '{args.service}' is properly set.")
            sys.exit(0)
        else:
            sys.exit(1)
    
    if not args.path:
        parser.error("the following arguments are required: path")
    
    # Check and prompt for API configuration if needed
    if not check_api_config(args.service):
        if not prompt_and_save_api_config(args.service):
            print("Translation cancelled due to missing API configuration.", file=sys.stderr)
            sys.exit(1)
    
    print("Loading model...")
    try:
        model = OnnxModel.load_available()
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
    
    input_path = Path(args.path)
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    params = {
        'lang_in': args.lang_in,
        'lang_out': args.lang_out,
        'service': args.service,
        'thread': args.thread,
        'model': model
    }
    
    if input_path.is_file():
        if not input_path.suffix.lower() == '.pdf':
            print(f"Error: Not a PDF file: {input_path}", file=sys.stderr)
            sys.exit(1)
        
        if is_translated_file(input_path) or has_translated_versions(input_path):
            print("File is already translated or translated versions exist")
            sys.exit(0)
        
        print(f"Translating: {input_path.name}")
        if translate_pdf(input_path, params):
            print(f"✓ Saved: {input_path.stem}-mono.pdf, {input_path.stem}-dual.pdf")
        else:
            sys.exit(1)
    
    elif input_path.is_dir():
        print(f"Scanning directory: {input_path}")
        pdfs = find_untranslated_pdfs(input_path)
        
        if not pdfs:
            print("No PDF files found that need translation")
            sys.exit(0)
        
        print(f"\nFound {len(pdfs)} file(s) to translate:")
        for pdf in pdfs:
            print(f"  - {pdf}")
        
        if input(f"\nStart translating? (y/n): ").lower() != 'y':
            print("Cancelled")
            sys.exit(0)
        
        success = 0
        for i, pdf in enumerate(pdfs, 1):
            print(f"\n[{i}/{len(pdfs)}] Translating: {pdf.name}")
            if translate_pdf(pdf, params):
                success += 1
                print(f"  ✓ Saved: {pdf.stem}-mono.pdf, {pdf.stem}-dual.pdf")
        
        print(f"\n{'='*50}")
        print(f"Translation complete!")
        print(f"Success: {success}, Failed: {len(pdfs) - success}")
        print(f"{'='*50}")
    
    else:
        print(f"Error: Invalid path: {input_path}", file=sys.stderr)
        sys.exit(1)
