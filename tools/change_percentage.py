def delta_percent(original, new):
    if original == 0:
        return 0
    return ((new - original) / original) * 100

data = [
    (0.8077, 0.9091),
    (0.5072, 0.8398),
    (0.6231, 0.8730),
    (0.4666, 0.8162),
    (0.499, 0.640)
]

for x in data:
    print(f"Original: {x[0]}, New: {x[1]}, Delta Percent: {delta_percent(x[0], x[1]):.2f}%")