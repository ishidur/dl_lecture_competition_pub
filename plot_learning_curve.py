import re
import numpy as np
from matplotlib import pyplot as plt


ptrn = re.compile(r"Epoch (\d+), Loss: ([\d\.]+),.*\n")
ptrn2 = re.compile(r"TestLoss: ([\d\.]+), Best:.*\n")


with open("log.txt", "r") as f:
    lns = "".join(f.readlines())
    recs = []
    m = ptrn.findall(lns)
    m2 = ptrn2.findall(lns)
    for _m, _m2 in zip(m, m2):
        ep, tlss = _m
        recs.append([int(ep), float(tlss), float(_m2)])
res = np.array(recs)
print(res)
plt.figure()
plt.plot(res[:, 0], res[:, 1], label="Train")
plt.plot(res[:, 0], res[:, 2], label="Validation")
plt.grid()
plt.yscale("log")
plt.ylim(top=1e1)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("log.png")
# plt.show()