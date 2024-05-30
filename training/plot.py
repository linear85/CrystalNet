import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

data_10FCV = pd.read_excel("10Fold_CV.xlsx")
data_large = pd.read_excel("largeSampleValidation.xlsx")

fig = plt.figure()
gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

plt.subplot(gs[0, 0])
plt.bar(data_10FCV['models'], data_10FCV['test_MAE_mean'], yerr=data_10FCV['test_std'], capsize=2)
plt.ylabel("MAE (eV/atom)")
plt.title("10Fold_CV")
plt.xticks(rotation=45)

plt.subplot(gs[0, 1])
plt.bar(data_large['models'], data_large['test_MAE_mean'], yerr=data_large['test_std'], capsize=2)
plt.ylabel("MAE (eV/atom)")
plt.title("LargeSample Validation")
plt.ylim(0, 1)
plt.xticks(rotation=45)

plt.subplots_adjust(top=0.95, bottom=0.14, left=0.1, right=0.99, hspace=0.25, wspace=0.3)
plt.gcf().set_size_inches(10, 5)
plt.show()