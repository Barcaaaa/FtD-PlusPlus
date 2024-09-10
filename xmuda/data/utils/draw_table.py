import matplotlib.pyplot as plt
import numpy as np


methods = [chr(9312), chr(9313), chr(9314)]

params = [0.69, 0.05, 0.105]
flops = [13.65, 0.95, 3.88]
gpu_mem = [14.849, 1.193, 2.795]
miou = [55.1, 56.8, 60.0]

# 创建一个2x2的子图布局
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# 设置每个子图的数据
axs[0, 0].bar(methods, params, color='lightcoral', width=0.5)
axs[0, 0].set_facecolor('#f0f0f0')
axs[0, 0].set_ylabel('Params (M)', fontsize=15)
axs[0, 0].set_ylim([0.0, 0.8])
axs[0, 0].tick_params(axis='x', which='major', labelsize=18)
axs[0, 0].tick_params(axis='y', which='major', labelsize=14)
for i, v in enumerate(params):
    axs[0, 0].text(i, v + 0.01, str(v), ha='center', va='bottom', fontsize=14)

axs[0, 1].bar(methods, flops, color='orange', width=0.5)
axs[0, 1].set_facecolor('#f0f0f0')
axs[0, 1].set_ylabel('FLOPs (G)', fontsize=15)
axs[0, 1].set_ylim([0, 16])
axs[0, 1].tick_params(axis='x', which='major', labelsize=18)
axs[0, 1].tick_params(axis='y', which='major', labelsize=14)
for i, v in enumerate(flops):
    axs[0, 1].text(i, v + 0.25, str(v), ha='center', va='bottom', fontsize=14)

axs[1, 0].bar(methods, gpu_mem, color='turquoise', width=0.5)
axs[1, 0].set_facecolor('#f0f0f0')
axs[1, 0].set_ylabel('GPU Memory (GB)', fontsize=15)
axs[1, 0].set_ylim([0, 18])
axs[1, 0].tick_params(axis='x', which='major', labelsize=18)
axs[1, 0].tick_params(axis='y', which='major', labelsize=14)
for i, v in enumerate(gpu_mem):
    axs[1, 0].text(i, v + 0.3, str(v), ha='center', va='bottom', fontsize=14)

axs[1, 1].bar(methods, miou, color='yellowgreen', width=0.5)
axs[1, 1].set_facecolor('#f0f0f0')
axs[1, 1].set_ylabel('mIoU (%)', fontsize=15)
axs[1, 1].set_ylim([25, 65])
axs[1, 1].tick_params(axis='x', which='major', labelsize=18)
axs[1, 1].tick_params(axis='y', which='major', labelsize=14)
for i, v in enumerate(miou):
    axs[1, 1].text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=14)

plt.tight_layout()

plt.savefig('complexity.png')
plt.savefig('complexity.pdf', format='pdf')


categories = ['2D', '3D', 'xM']
values_1 = [57.1, 70.4, 68.3]
values_2 = [68.7, 68.9, 68.9]
values_3 = [56.8, 68.7, 67.7]
values_4 = [68.8, 69.6, 71.0]

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('#f0f0f0')

colors = ['steelblue', 'slateblue', 'grey', 'mediumvioletred']
for idx, category in enumerate(categories):
    ax.bar(idx+1-0.3, values_1[idx], color=colors[0], label='2D/3D \u2192 Fus' if idx == 0 else "_nolegend_", width=0.2)
    ax.text(idx+1-0.3, values_1[idx]+0.3, str(values_1[idx]), ha='center', va='bottom', fontsize=14)
    ax.bar(idx+1-0.1, values_2[idx], color=colors[1], label='2D+Fus \u21C4 3D+Fus' if idx == 0 else "_nolegend_", width=0.2)
    ax.text(idx+1-0.1, values_2[idx]+0.3, str(values_2[idx]), ha='center', va='bottom', fontsize=14)
    ax.bar(idx+1+0.1, values_3[idx], color=colors[2], label='2D \u21C4 3D+Fus' if idx == 0 else "_nolegend_", width=0.2)
    ax.text(idx+1+0.1, values_3[idx]+0.3, str(values_3[idx]), ha='center', va='bottom', fontsize=14)
    ax.bar(idx+1+0.3, values_4[idx], color=colors[3], label='2D+Fus \u21C4 3D' if idx == 0 else "_nolegend_", width=0.2)
    ax.text(idx+1+0.3, values_4[idx]+0.3, str(values_4[idx]), ha='center', va='bottom', fontsize=14)

ax.set_xticks([1, 2, 3], categories)
ax.tick_params(axis='x', which='major', labelsize=14)
ax.tick_params(axis='y', which='major', labelsize=14)
ax.set_ylim([50, 75])
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_ylabel('mIoU (%)', fontsize=14)

# ax.legend()
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, shadow=True, ncol=4, fontsize=14)

plt.tight_layout()

plt.savefig('direction.png')
plt.savefig('direction.pdf', format='pdf')
