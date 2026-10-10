from llama_cpp import Llama
import argparse
import sys
import os

# Configuration - Edit these defaults if needed
DEFAULT_MODEL = r"D:\Mobile Ai\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
DEFAULT_CTX = 32768
DEFAULT_THREADS = 8
DEFAULT_GPU_LAYERS = 40

def main():
    parser = argparse.ArgumentParser(
        description="Academic Lecture Summarizer - Simply provide your lecture file path",
        usage="%(prog)s [LECTURE_FILE_PATH]"
    )
    parser.add_argument(
        'lecture_path',
        type=str,
        help='Path to lecture text file (required)'
    )
    
    args = parser.parse_args()

    # Validate file existence
    if not os.path.exists(args.lecture_path):
        print(f"Error: File not found - {args.lecture_path}")
        sys.exit(1)

    try:
        # Initialize model
        llm = Llama(
            model_path=DEFAULT_MODEL,
            n_ctx=DEFAULT_CTX,
            n_threads=DEFAULT_THREADS,
            n_gpu_layers=DEFAULT_GPU_LAYERS,
            verbose=False
        )

        # Read lecture content
        with open(args.lecture_path, 'r', encoding='utf-8') as f:
            content = f.read()

        prompt = f"""\
    [INST] <<SYS>>
    Generate a **highly accurate and structured** summary of the lecture below. 
    - **Preserve all key concepts, formulas, and explanations.**  
    - **Do not add incorrect information or generalizations.**  
    - If the lecture contains numerical or visual demonstrations, **summarize them in words.**  
    - Keep the logical flow and ensure no important details are omitted.
    <</SYS>>

{content}[/INST]"""

        result = llm.create_completion(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.1
        )['choices'][0]['text'].strip()

        # Display output
        print("\n" + "="*50)
        print("LECTURE SUMMARY:")
        print("="*50)
        print(result + "\n")

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()