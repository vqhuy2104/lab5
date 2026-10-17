# LAB 5 - BAI 4: BINARY CROSS-ENTROPY VA LAPLACE SMOOTHING
import math

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    # Gioi han xac suat de tranh log(0)
    y_pred = max(eps, min(1 - eps, y_pred))

    # BCE = -(y*log(p) + (1-y)*log(1-p))
    loss = -(y_true * math.log(y_pred) +
             (1 - y_true) * math.log(1 - y_pred))
    return loss


def laplace_smoothing(count_word, count_class, vocab_size, alpha=1.0):
    # Cong alpha vao tu so va alpha * kich thuoc tu dien vao mau so
    denominator = count_class + alpha * vocab_size
    if denominator == 0:
        raise ValueError("Mau so bang 0; hay kiem tra tham so dau vao.")
    return (count_word + alpha) / denominator


if __name__ == "__main__":
    print(f"Loss khi du doan chuan (y=1, p=0.9): {binary_cross_entropy(1, 0.9):.4f}")
    print(f"Loss khi du doan sai (y=1, p=0.1): {binary_cross_entropy(1, 0.1):.4f}")

    p_smoothed = laplace_smoothing(
        count_word=0, count_class=100, vocab_size=500
    )
    print(f"Xac suat sau lam muot: {p_smoothed:.4f}")
    # Ket qua gan dung: 0.1054, 2.3026 va 0.0017
