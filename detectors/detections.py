import json
import numpy as np
import pandas as pd

def positive(x):
    return x if x >= 0 else 0

with open('detections.json') as f:
  detections = json.load(f)

for img in detections:
    filename = img["filename"].split("/")[-1].split(".")[0] + ".txt"
    rows = []
    for head in img["annotations"]:
        h = head["height"]
        w = head["width"]
        x = head["x"]
        y = head["y"]

        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        row = [positive(xmin), positive(ymin), positive(xmax), positive(ymax)]
        rows.append(row)
    df = pd.DataFrame(rows)
    np.savetxt(filename, df.values, fmt='%d')

