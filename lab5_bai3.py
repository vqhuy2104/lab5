# LAB 5 - BAI 3: NAIVE BAYES VOI DAC TRUNG NHI PHAN
def naive_bayes_predict(dataset, x_new):
    # Dem so mau thuoc moi lop
    class_counts = {0: 0, 1: 0}
    feature_ones = {0: [0] * len(x_new), 1: [0] * len(x_new)}

    for features, label in dataset:
        class_counts[label] += 1
        for i, value in enumerate(features):
            if value == 1:
                feature_ones[label][i] += 1

    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset khong duoc rong.")

    scores = {}
    for label in [0, 1]:
        # Xac suat tien nghiem P(Y)
        prior = class_counts[label] / total
        score = prior

        # Laplace smoothing cho dac trung nhi phan:
        # P(x_i=1|Y) = (so lan x_i=1 + 1) / (so mau lop Y + 2)
        # P(x_i=0|Y) = (so lan x_i=0 + 1) / (so mau lop Y + 2)
        for i, value in enumerate(x_new):
            count_one = feature_ones[label][i]
            count_class = class_counts[label]
            p_one = (count_one + 1) / (count_class + 2)
            p_zero = (count_class - count_one + 1) / (count_class + 2)
            score *= p_one if value == 1 else p_zero

        scores[label] = score

    # Tra ve nhan co diem hau nghiem cao hon
    return 1 if scores[1] > scores[0] else 0


if __name__ == "__main__":
    dataset = [
        ([1, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1),
        ([0, 1], 0),
        ([0, 0], 0),
        ([0, 0], 0),
    ]

    x_new = [1, 0]
    prediction = naive_bayes_predict(dataset, x_new)
    print("Mau moi:", x_new)
    print("Nhan du doan:", prediction)
    print("Ket luan:", "Spam" if prediction == 1 else "Thuong")
    # Ket qua mong doi theo de bai: 1 (Spam)
