"""Check repository Markdown links and documented source-file references offline."""

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    failures = []
    count = 0
    for name in sorted(set(names)):
        path = ROOT / name
        if path.suffix != ".md" or not path.is_file():
            continue
        count += 1
        text = path.read_text()
        for target in re.findall(r"\[[^\]\n]*\]\(([^)\s]+)\)", text):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            if not destination.is_relative_to(ROOT) or not destination.exists():
                failures.append(f"{name}: missing or external local link {target}")
        for target in re.findall(r"`([^`\n]+)`", text):
            if (
                target.startswith(("docs/", "configs/", "scripts/", "engine/", "training/"))
                and " " not in target
                and not any(c in target for c in "*<>|")
                and not (ROOT / target.split("#")[0]).exists()
            ):
                failures.append(f"{name}: missing repository reference {target}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Documentation links and source references passed ({count} Markdown files)")


if __name__ == "__main__":
    main()
