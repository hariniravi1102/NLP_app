import os
import json
with open("feedback_data.jsonl", "a", encoding="utf-8") as f:
    json.dump({"test": "value"}, f)
    f.write("\n")