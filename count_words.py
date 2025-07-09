import sys
from pathlib import Path


def count_words_in_file(file_path: str) -> int:
    """Return the number of words in the given text file."""
    text = Path(file_path).read_text(encoding='utf-8')
    words = text.split()
    return len(words)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python count_words.py <file>")
        return 1
    file_path = argv[0]
    count = count_words_in_file(file_path)
    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
