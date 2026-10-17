# LAB 5 - BAI 2: INFORMATION GAIN
import math

def compute_entropy(labels):
    if not labels:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    entropy = 0.0
    for count in counts.values():
        p = count / len(labels)
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_information_gain(parent_labels, subsets_labels):
    # Entropy cua tap cha
    if not parent_labels:
        return 0.0

    h_parent = compute_entropy(parent_labels)
    total_len = len(parent_labels)

    # Entropy co dieu kien sau khi chia thanh cac tap con
    h_children = 0.0
    for sub in subsets_labels:
        if len(sub) > 0:
            weight = len(sub) / total_len
            h_children += weight * compute_entropy(sub)

    # IG = entropy tap cha - entropy trung binh co trong so cua cac tap con
    return round(h_parent - h_children, 4)


if __name__ == "__main__":
    parent = ["Yes", "Yes", "No", "No"]
    sub1 = ["Yes", "Yes"]
    sub2 = ["No", "No"]

    print("Information Gain:", compute_information_gain(parent, [sub1, sub2]), "bits")
    # Ket qua mong doi: 1.0 bits
