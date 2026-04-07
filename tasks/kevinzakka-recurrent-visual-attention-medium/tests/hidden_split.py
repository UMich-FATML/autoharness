HIDDEN_SPECS = [
    {"digit": digit, "pos_idx": pos_idx, "seed": 19000 + digit * 1000 + pos_idx * 50 + repeat}
    for digit in range(10)
    for pos_idx in range(4)
    for repeat in range(6)
]
