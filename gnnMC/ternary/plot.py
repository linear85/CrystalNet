import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpltern


####################################################################################################
# # read experimental data

def read_name(name: str) -> list[int]:
    pos_1 = name.index("_")
    c_1 = int(name[:pos_1])/100
    name = name[pos_1+1:]
    pos_2 = name.index("_")
    c_2 = int(name[:pos_2])/100
    name = name[pos_1+1:]
    pos_3 = name.index("_")
    c_3 = int(name[:pos_3])/100
    return [c_1, c_2, c_3]


xls = pd.ExcelFile('ternary_performance.xlsx')
random_data = []
order_data = []
for sheet in xls.sheet_names:
    cur_df = pd.read_excel(xls, sheet_name=sheet)
    comp = read_name(sheet)
    cur_random = comp + [cur_df["Diff"][0]]
    cur_order = comp + [cur_df["Diff"][1]]
    random_data.append(tuple(cur_random))
    order_data.append(tuple(cur_order))

random_data = np.array(random_data)
order_data  = np.array(order_data)

# ==========================================================================================================================


label_size = 14
legend_size = 12
title_size = 16

fig = plt.figure(figsize=(11.5, 5.5))
fig.subplots_adjust(left=0.02, right=0.88, wspace=0.35)

ax = fig.add_subplot(1, 2, 1, projection="ternary")
pc = ax.scatter(random_data[:, 0], 
                random_data[:, 1], 
                random_data[:, 2], 
                c=random_data[:, 3]*1000, 
                # marker='^', 
                s=60)
ax.grid(linestyle='--')
ax.set_tlabel("Mo", fontsize=label_size)
ax.set_llabel("Nb", fontsize=label_size)
ax.set_rlabel("Ta", fontsize=label_size)
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")
plt.title("Random", pad=30, fontsize=title_size)

cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(pc, cax=cax)
colorbar.set_label("MAE (meV/atom)", rotation=270, va="baseline", fontsize=legend_size)

ax = fig.add_subplot(1, 2, 2, projection="ternary")
pc = ax.scatter(order_data[:, 0], 
                order_data[:, 1], 
                order_data[:, 2], 
                c=order_data[:, 3]*1000,
                # marker='^', 
                s=60)
ax.grid(linestyle='--')
ax.set_tlabel("Mo", fontsize=label_size)
ax.set_llabel("Nb", fontsize=label_size)
ax.set_rlabel("Ta", fontsize=label_size)
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")
plt.title("Order", pad=30, fontsize=title_size)

cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(pc, cax=cax)
colorbar.set_label("MAE (meV/atom)", rotation=270, va="baseline", fontsize=legend_size)

plt.savefig("Ternary_Performance.png", dpi=1200)
plt.show()