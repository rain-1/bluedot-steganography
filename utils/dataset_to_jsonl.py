from datasets import load_from_disk
import json
import sys
dataset = load_from_disk(sys.argv[1])
try:
    for row in dataset:
        sys.stdout.write(json.dumps(row) + "\n")
        sys.stdout.flush() 
except BrokenPipeError:
    sys.stderr.close()
