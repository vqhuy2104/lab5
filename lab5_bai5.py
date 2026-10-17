# LAB 5 - BAI 5: PIPELINE LOC EMAIL SPAM VA CONFUSION MATRIX
import math

def extract_vocabulary(emails):
    # Gom tat ca tu duy nhat trong cac email huan luyen
    vocabulary = set()
    for words, label in emails:
        vocabulary.update(words)
    return vocabulary


def train_naive_bayes(train_emails, alpha=1.0):
    # Moi email co dang (danh_sach_tu, nhan), nhan 1=Spam, 0=Ham
    vocabulary = extract_vocabulary(train_emails)
    vocab_size = len(vocabulary)

    class_doc_counts = {0: 0, 1: 0}
    word_counts = {0: {}, 1: {}}
    total_words = {0: 0, 1: 0}

    for words, label in train_emails:
        class_doc_counts[label] += 1
        for word in words:
            word_counts[label][word] = word_counts[label].get(word, 0) + 1
            total_words[label] += 1

    total_docs = len(train_emails)
    if total_docs == 0:
        raise ValueError("Du lieu huan luyen khong duoc rong.")

    # Xac suat tien nghiem theo so email cua moi lop
    priors = {
        label: class_doc_counts[label] / total_docs
        for label in [0, 1]
    }

    # Xac suat tu co Laplace smoothing:
    # P(word|class) = (count(word,class)+alpha)/(tong so tu cua class + alpha*|V|)
    likelihoods = {0: {}, 1: {}}
    for label in [0, 1]:
        denominator = total_words[label] + alpha * vocab_size
        for word in vocabulary:
            count = word_counts[label].get(word, 0)
            likelihoods[label][word] = (count + alpha) / denominator

    return {
        "vocabulary": vocabulary,
        "priors": priors,
        "likelihoods": likelihoods,
        "total_words": total_words,
        "alpha": alpha,
    }


def predict_email(email_words, model):
    # Dung log-likelihood: log(P(Y)) + sum(log(P(word|Y))).
    # Nhan nhieu xac suat nho co the gay underflow (tich bi lam tron ve 0).
    # Cong log giup thay tich cac so nho bang tong log, on dinh hon.
    scores = {}
    vocabulary = model["vocabulary"]
    priors = model["priors"]
    likelihoods = model["likelihoods"]

    for label in [0, 1]:
        score = math.log(priors[label])
        for word in email_words:
            # Tu ngoai tu dien duoc xu ly bang xac suat Laplace tuong ung.
            if word in vocabulary:
                probability = likelihoods[label][word]
            else:
                # Mot tu chua thay trong tu dien khong the tra bang truc tiep.
                # Dung xac suat nho theo cong thuc Laplace cho mot tu moi.
                vocab_size = len(vocabulary)
                # Xac suat cho tu chua thay: alpha / (tong tu lop + alpha*(|V|+1))
                # Can tong so tu theo lop, duoc luu trong model.
                total_class_words = model["total_words"][label]
                alpha = model["alpha"]
                probability = alpha / (total_class_words + alpha * (vocab_size + 1))
            score += math.log(probability)
        scores[label] = score

    return 1 if scores[1] > scores[0] else 0


def evaluate_confusion_matrix(y_true, y_pred):
    # Quy uoc lop 1 la Spam, lop 0 la Ham
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


if __name__ == "__main__":
    train_emails = [
        (["free", "money", "offer"], 1),
        (["free", "ticket", "win"], 1),
        (["money", "bonus", "free"], 1),
        (["meeting", "project", "deadline"], 0),
        (["project", "report", "review"], 0),
        (["schedule", "meeting", "room"], 0),
    ]

    model = train_naive_bayes(train_emails)
    print("Vocabulary:", sorted(model["vocabulary"]))
    print("P(Spam):", model["priors"][1])
    print("P(Ham):", model["priors"][0])

    test_email = ["free", "project", "bonus"]
    prediction = predict_email(test_email, model)
    print("Email test:", test_email)
    print("Nhan du doan:", prediction, "-", "Spam" if prediction == 1 else "Ham")

    # Danh gia minh hoa tren tap kiem tra nho; co the thay bang du lieu test cua ban.
    test_emails = [
        (["free", "money"], 1),
        (["meeting", "project"], 0),
        (["free", "bonus"], 1),
        (["report", "review"], 0),
    ]
    y_true = [label for words, label in test_emails]
    y_pred = [predict_email(words, model) for words, label in test_emails]
    print("Nhan that:", y_true)
    print("Nhan du doan:", y_pred)
    print("Confusion matrix:", evaluate_confusion_matrix(y_true, y_pred))
    # TP: Spam du doan dung; TN: Ham du doan dung
    # FP: Ham bi du doan nham thanh Spam; FN: Spam bi bo sot thanh Ham
