#!/usr/bin/env sh
# Set up the Research Paper Answer Bot. All logic is in backend/scripts/setup.py;
# run "./setup.sh --help" for the options.
cd "$(dirname "$0")" || exit 1

for py in python3.12 python3 python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))"; then
        exec "$py" backend/scripts/setup.py "$@"
    fi
done
echo "Python 3.12 was not found. Install it from https://www.python.org/downloads/ and rerun ./setup.sh."
exit 1
