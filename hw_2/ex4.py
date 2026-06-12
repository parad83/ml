import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

RANDOM_SEED = 42
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)

svm = SVC(kernel="rbf", gamma=0.0012, C=0.85)
svm.fit(X_train, y_train)
acc_svm = svm.score(X_test, y_test)

n_sv_per_class = svm.n_support_

starts = np.concatenate(([0], np.cumsum(n_sv_per_class)))
idx0 = np.arange(starts[0], starts[1])
idx1 = np.arange(starts[1], starts[2])

coef0 = svm.dual_coef_[0, idx0] 
coef1 = svm.dual_coef_[0, idx1]  

active0 = np.abs(coef0) > 1e-12
active1 = np.abs(coef1) > 1e-12
n_v = active0.sum() + active1.sum()

sv0 = svm.support_vectors_[idx0][active0]
sv1 = svm.support_vectors_[idx1][active1]
a0 = np.abs(coef0[active0])
a1 = np.abs(coef1[active1])

top0 = np.argsort(a0)[::-1][:4]
top1 = np.argsort(a1)[::-1][:4]

fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
for col, i in enumerate(top0):
    ax = axes[0, col]
    ax.imshow(sv0[i].reshape(8, 8), cmap="gray_r", interpolation="nearest")
    ax.set_title(f"class 0\n|coef| = {a0[i]:.3f}")
    ax.axis("off")
for col, i in enumerate(top1):
    ax = axes[1, col]
    ax.imshow(sv1[i].reshape(8, 8), cmap="gray_r", interpolation="nearest")
    ax.set_title(f"class 1\n|coef| = {a1[i]:.3f}")
    ax.axis("off")
fig.tight_layout()
fig.savefig("./sv_0_vs_1.png", dpi=150, bbox_inches="tight")