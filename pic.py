# Modified code with specific colors for MixATIS and MixSNIPS as provided earlier
import matplotlib.pyplot as plt

# Define the data
metrics = ['Intent Accuracy', 'Slot F1 Score', 'Overall Accuracy']
MixATIS_values = [21.62, 5.91, 0.36]
MixSNIPS_values = [71.49, 3.71, 0.09]

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Set the size of the figure
fig.set_size_inches(8, 4.5)

# Set the bar width
bar_width = 0.35

# Set the opacity
opacity = 0.8

# Set the index for the groups
index = range(len(metrics))

# Define the specific colors
color_mixatis = (122/255, 184/255, 242/255)  # MixATIS RGB color converted to [0, 1] scale
color_mixsnips = (173/255, 203/255, 227/255)  # MixSNIPS RGB color converted to [0, 1] scale

# Plot the data
rects1 = ax.bar(index, MixATIS_values, bar_width,
                alpha=opacity, color=color_mixatis, label='MixATIS')

rects2 = ax.bar([i + bar_width for i in index], MixSNIPS_values, bar_width,
                alpha=opacity, color=color_mixsnips, label='MixSNIPS')

# Add the text for the labels, title and axes ticks
ax.set_xlabel('Metrics', fontsize=13)
ax.set_ylabel('Performance (%)', fontsize=13)
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(metrics, fontsize=13)
ax.legend()

# Set the y-axis ticks font size
plt.yticks(fontsize=15)

# Add the percentage values on top of the bars
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate('{:.2f}%'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=13)

# Save the figure as a PDF
pdf_path = 'ChatGPT_new_1.pdf'
plt.tight_layout()
plt.savefig(pdf_path, format='pdf')

# Show the plot
plt.show()

# Return the path to the saved PDF
pdf_path
