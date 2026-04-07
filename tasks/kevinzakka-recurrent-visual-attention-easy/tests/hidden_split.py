HIDDEN_SPECS = [
    {"digit": digit, "pos_idx": pos_idx, "seed": 9100 + digit * 100 + pos_idx * 10 + repeat}
    for digit in range(10)
    for pos_idx in range(9)
    for repeat in range(3)
]
