# LAB 5 - BAI 1: TINH SHANNON ENTROPY
import math

def compute_entropy(labels):
    # Neu danh sach rong thi entropy bang 0
    if not labels:
        return 0.0

    total_samples = len(labels)

    # Dem so lan xuat hien cua tung nhan
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Tinh Shannon Entropy: H(D) = -sum(p * log2(p))
    entropy = 0.0
    for count in label_counts.values():
        p = count / total_samples
        if p > 0:
            entropy -= p * math.log2(p)

    return round(entropy, 4)


if __name__ == "__main__":
    labels_mixed = ["Yes", "Yes", "No", "Yes", "No"]
    labels_pure = ["Yes", "Yes", "Yes"]

    print("Entropy tap hon loan:", compute_entropy(labels_mixed), "bits")
    print("Entropy tap tinh khiet:", compute_entropy(labels_pure), "bits")
    # Ket qua mong doi: 0.971 va 0.0
