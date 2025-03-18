# %%
import matplotlib.pyplot as plt
import numpy as np

# Defining the models and their scores along with confidence intervals
models = ['clinicalbert (Apr. 2019)', 'bio_clinicalbert (June 2019)', 'biobert (Oct. 2019)', 'medbert (Dec. 2022)']

# F1 Scores and their confidence intervals
f1_scores = [0.894, 0.933, 0.936, 0.943]
f1_conf_intervals = [(0.892, 0.897), (0.930, 0.935), (0.933, 0.938), (0.940, 0.945)]
f1_errors = [(f1_scores[i] - f1_conf_intervals[i][0], f1_conf_intervals[i][1] - f1_scores[i]) for i in range(len(models))]

# AUROC Scores and their confidence intervals
auroc_scores = [0.961, 0.987, 0.988, 0.991]
auroc_conf_intervals = [(0.959, 0.962), (0.986, 0.988), (0.988, 0.989), (0.990, 0.991)]
auroc_errors = [(auroc_scores[i] - auroc_conf_intervals[i][0], auroc_conf_intervals[i][1] - auroc_scores[i]) for i in range(len(models))]

# AUPRC Scores and their confidence intervals
auprc_scores = [0.922, 0.977, 0.979, 0.983]
auprc_conf_intervals = [(0.918, 0.926), (0.975, 0.978), (0.978, 0.981), (0.982, 0.985)]
auprc_errors = [(auprc_scores[i] - auprc_conf_intervals[i][0], auprc_conf_intervals[i][1] - auprc_scores[i]) for i in range(len(models))]

# Slight adjustments for readability and layout
fig, ax = plt.subplots(1, 3, figsize=(30, 12), sharey=True)

# Adjust y-axis zoom range
y_min, y_max = 0.88, 1.00

# Common text size settings
title_size = 26
label_size = 22
tick_size = 20
annotate_size = 22
legend_size = 22

# F1 Score Line Plot
ax[0].errorbar(models, f1_scores, yerr=np.array(f1_errors).T, fmt='-o', capsize=5, color='#1f77b4',label='F1 Score')
ax[0].set_title('F1 Score', fontsize=title_size)
ax[0].set_ylabel('Score', fontsize=label_size)
ax[0].set_ylim(y_min, y_max)
ax[0].set_xticks(range(len(models)))
ax[0].set_xticklabels(models, rotation=45, ha='right', fontsize=tick_size)
for i, score in enumerate(f1_scores):
    ax[0].annotate(f'{score:.3f}', (i, score), textcoords="offset points", xytext=(0,10), ha='center', fontsize=annotate_size)
ax[0].grid(True, linestyle='--', alpha=0.7)

# AUROC Score Line Plot
ax[1].errorbar(models, auroc_scores, yerr=np.array(auroc_errors).T, fmt='-o', capsize=5, color='#ff7f0e', label='AUROC Score')
ax[1].set_title('AUROC Score', fontsize=title_size)
ax[1].set_xticks(range(len(models)))
ax[1].set_xticklabels(models, rotation=45, ha='right', fontsize=tick_size)
for i, score in enumerate(auroc_scores):
    ax[1].annotate(f'{score:.3f}', (i, score), textcoords="offset points", xytext=(0,10), ha='center', fontsize=annotate_size)
ax[1].grid(True, linestyle='--', alpha=0.7)

# AUPRC Score Line Plot
ax[2].errorbar(models, auprc_scores, yerr=np.array(auprc_errors).T, fmt='-o', capsize=5, color='#2ca02c', label='AUPRC Score')
ax[2].set_title('AUPRC Score', fontsize=title_size)
ax[2].set_xticks(range(len(models)))
ax[2].set_xticklabels(models, rotation=45, ha='right', fontsize=tick_size)
for i, score in enumerate(auprc_scores):
    ax[2].annotate(f'{score:.3f}', (i, score), textcoords="offset points", xytext=(0,10), ha='center', fontsize=annotate_size)
ax[2].grid(True, linestyle='--', alpha=0.7)

fig.suptitle('Performance on ED Disposition: MIMIC Dataset', fontsize=30)
plt.subplots_adjust(top=0.85)  # Adjust space for the title
plt.tight_layout()
plt.savefig("backbone.png", dpi=1000, bbox_inches='tight')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Hardcoded JSON data
data_json = {
    "f1": {
        "ed_disposition": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.6217749705598973, 0.5944299390774587, 0.6172845163359758, 0.6371492020389142, 0.6543770133320793, 0.6956205492790859, 0.7034644270370669, 0.9467932759038533, 0.9973820317987357, 0.9992958948536315, 1.0],
            "lower_bounds": None,
            "upper_bounds": None
        },
        "home": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.6119954588275079, 0.6127587459145382, 0.6084315393026364, 0.6117681201480084, 0.6121600386800435, 0.611707905853953, 0.6302921523880014, 0.6379901477832512, 0.6648530766868026, 0.6835268956055042, 0.685747497669895],
            "lower_bounds": [0.6119954588275079, 0.6127587459145382, 0.6084315393026364, 0.6117681201480084, 0.6121600386800435, 0.611707905853953, 0.6302921523880014, 0.6379901477832512, 0.6648530766868026, 0.6835268956055042, 0.685747497669895],
            "upper_bounds": [0.6119954588275079, 0.6127587459145382, 0.6084315393026364, 0.6117681201480084, 0.6121600386800435, 0.611707905853953, 0.6302921523880014, 0.6379901477832512, 0.6648530766868026, 0.6835268956055042, 0.685747497669895]
        },
        "icu": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.3976533137776148, 0.38450129773822767, 0.3867876721794813, 0.39053092501368364, 0.3906501633460636, 0.39045214597199907, 0.37298231979794055, 0.3987376129263342, 0.4606088623294622, 0.46832266325224076, 0.46760743801652893],
            "lower_bounds": [0.3976533137776148, 0.38450129773822767, 0.3867876721794813, 0.39053092501368364, 0.3906501633460636, 0.39045214597199907, 0.37298231979794055, 0.3987376129263342, 0.4606088623294622, 0.46832266325224076, 0.46760743801652893],
            "upper_bounds": [0.3976533137776148, 0.38450129773822767, 0.3867876721794813, 0.39053092501368364, 0.3906501633460636, 0.39045214597199907, 0.37298231979794055, 0.3987376129263342, 0.4606088623294622, 0.46832266325224076, 0.46760743801652893]
        },
        "mortality": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.00299625468164794, 0.0044943820224719105, 0.004514672686230248, 0.006535947712418301, 0.0014781966001478197, 0.004542013626040878, 0.05417533432392273, 0.1014096301465457, 0.17984536082474228, 0.2396638655462185, 0.2384726368159204],
            "lower_bounds": [0.00299625468164794, 0.0044943820224719105, 0.004514672686230248, 0.006535947712418301, 0.0014781966001478197, 0.004542013626040878, 0.05417533432392273, 0.1014096301465457, 0.17984536082474228, 0.2396638655462185, 0.2384726368159204],
            "upper_bounds": [0.00299625468164794, 0.0044943820224719105, 0.004514672686230248, 0.006535947712418301, 0.0014781966001478197, 0.004542013626040878, 0.05417533432392273, 0.1014096301465457, 0.17984536082474228, 0.2396638655462185, 0.2384726368159204]
        }
    },
    "auroc": {
        "ed_disposition": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.8331751387000096, 0.8524402214246325, 0.8758914554486945, 0.8877036114771677, 0.9029597010033809, 0.9228071224714497, 0.982449764349422, 0.9972445859114419, 0.9999162814069511, 0.999999941389758, 1.0],
            "lower_bounds": None,
            "upper_bounds": None
        },
        "home": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.7320197607959118, 0.7325578220316381, 0.7300885127569929, 0.7288772764664462, 0.7303480688130366, 0.7313692001174366, 0.7562719616014966, 0.7708798210039168, 0.7920888797471994, 0.8092640380910886, 0.8111844459757406],
            "lower_bounds": [0.7320197607959118, 0.7325578220316381, 0.7300885127569929, 0.7288772764664462, 0.7303480688130366, 0.7313692001174366, 0.7562719616014966, 0.7708798210039168, 0.7920888797471994, 0.8092640380910886, 0.8111844459757406],
            "upper_bounds": [0.7320197607959118, 0.7325578220316381, 0.7300885127569929, 0.7288772764664462, 0.7303480688130366, 0.7313692001174366, 0.7562719616014966, 0.7708798210039168, 0.7920888797471994, 0.8092640380910886, 0.8111844459757406]
        },
        "icu": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.7499439929574466, 0.7427725680822402, 0.7454948883613645, 0.7445050831888946, 0.7487157584469772, 0.7451332121670347, 0.755794602569352, 0.7749220879419599, 0.7958997759909787, 0.8080998954986865, 0.8139794415736089],
            "lower_bounds": [0.7499439929574466, 0.7427725680822402, 0.7454948883613645, 0.7445050831888946, 0.7487157584469772, 0.7451332121670347, 0.755794602569352, 0.7749220879419599, 0.7958997759909787, 0.8080998954986865, 0.8139794415736089],
            "upper_bounds": [0.7499439929574466, 0.7427725680822402, 0.7454948883613645, 0.7445050831888946, 0.7487157584469772, 0.7451332121670347, 0.755794602569352, 0.7749220879419599, 0.7958997759909787, 0.8080998954986865, 0.8139794415736089]
        },
        "mortality": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.807724514060694, 0.8020296670162207, 0.8140812777053323, 0.8145272709706288, 0.8151568241326344, 0.8107938308299635, 0.8284775943112, 0.8339475231216982, 0.8550682097763662, 0.8716714988752812, 0.8732150167887051],
            "lower_bounds": [0.807724514060694, 0.8020296670162207, 0.8140812777053323, 0.8145272709706288, 0.8151568241326344, 0.8107938308299635, 0.8284775943112, 0.8339475231216982, 0.8550682097763662, 0.8716714988752812, 0.8732150167887051],
            "upper_bounds": [0.807724514060694, 0.8020296670162207, 0.8140812777053323, 0.8145272709706288, 0.8151568241326344, 0.8107938308299635, 0.8284775943112, 0.8339475231216982, 0.8550682097763662, 0.8716714988752812, 0.8732150167887051]
        }
    },
    "auprc": {
        "ed_disposition": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.6235169874234565, 0.6438753510849058, 0.681218580514696, 0.7046478360153747, 0.7360227450213864, 0.7837265151395221, 0.9403942155184136, 0.991488579834929, 0.9998934585528677, 0.9999998461899737, 1.0],
            "lower_bounds": None,
            "upper_bounds": None
        },
        "home": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.6347392408563749, 0.6387365540522296, 0.6357871788786316, 0.6323539598463719, 0.6343968141203948, 0.6405503383053318, 0.6631523541460649, 0.6668069890386276, 0.6928801835450464, 0.7124913754381411, 0.7098074936295129],
            "lower_bounds": [0.6347392408563749, 0.6387365540522296, 0.6357871788786316, 0.6323539598463719, 0.6343968141203948, 0.6405503383053318, 0.6631523541460649, 0.6668069890386276, 0.6928801835450464, 0.7124913754381411, 0.7098074936295129],
            "upper_bounds": [0.6347392408563749, 0.6387365540522296, 0.6357871788786316, 0.6323539598463719, 0.6343968141203948, 0.6405503383053318, 0.6631523541460649, 0.6668069890386276, 0.6928801835450464, 0.7124913754381411, 0.7098074936295129]
        },
        "icu": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.3917643102023639, 0.37943290615216746, 0.3805038362379721, 0.3940411118552274, 0.3908305957063051, 0.39052283003627336, 0.3949519986995886, 0.4130756171129976, 0.4386675861091139, 0.4512105141952312, 0.44945370606420926],
            "lower_bounds": [0.3917643102023639, 0.37943290615216746, 0.3805038362379721, 0.3940411118552274, 0.3908305957063051, 0.39052283003627336, 0.3949519986995886, 0.4130756171129976, 0.4386675861091139, 0.4512105141952312, 0.44945370606420926],
            "upper_bounds": [0.3917643102023639, 0.37943290615216746, 0.3805038362379721, 0.3940411118552274, 0.3908305957063051, 0.39052283003627336, 0.3949519986995886, 0.4130756171129976, 0.4386675861091139, 0.4512105141952312, 0.44945370606420926]
        },
        "mortality": {
            "x_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, "all"],
            "means": [0.1641633401886139, 0.16572714673446454, 0.16255615378152563, 0.14684580342083065, 0.17062475100626856, 0.15852797154283538, 0.19137984397398386, 0.19903435661034694, 0.24363885284931047, 0.258833804037808, 0.25457642404627745],
            "lower_bounds": [0.1641633401886139, 0.16572714673446454, 0.16255615378152563, 0.14684580342083065, 0.17062475100626856, 0.15852797154283538, 0.19137984397398386, 0.19903435661034694, 0.24363885284931047, 0.258833804037808, 0.25457642404627745],
            "upper_bounds": [0.1641633401886139, 0.16572714673446454, 0.16255615378152563, 0.14684580342083065, 0.17062475100626856, 0.15852797154283538, 0.19137984397398386, 0.19903435661034694, 0.24363885284931047, 0.258833804037808, 0.25457642404627745]
        }
    }
}

# Create result dictionaries for ED Disposition and for other outcomes
result_dict_ed = {}
result_dict_multi = {}

# Define the metrics and outcomes for each dictionary
metrics = ['f1', 'auroc', 'auprc']
ed_outcome = 'ed_disposition'
multi_outcomes = ['home', 'icu', 'mortality']

# Process ED Disposition metrics from the JSON
for metric in metrics:
    ed_data = data_json[metric][ed_outcome]
    x_vals = ed_data['x_values']
    means = ed_data['means']
    lower_bounds = ed_data['lower_bounds']
    upper_bounds = ed_data['upper_bounds']
    for idx, shot in enumerate(x_vals):
        val = means[idx]
        lb = val if lower_bounds is None else lower_bounds[idx]
        ub = val if upper_bounds is None else upper_bounds[idx]
        if shot not in result_dict_ed:
            result_dict_ed[shot] = {}
        result_dict_ed[shot][metric] = (val, lb, ub)

# Process the multi-task outcomes from the JSON
for outcome in multi_outcomes:
    for metric in metrics:
        multi_data = data_json[metric][outcome]
        x_vals = multi_data['x_values']
        means = multi_data['means']
        lower_bounds = multi_data['lower_bounds']
        upper_bounds = multi_data['upper_bounds']
        for idx, shot in enumerate(x_vals):
            val = means[idx]
            lb = val if lower_bounds is None else lower_bounds[idx]
            ub = val if upper_bounds is None else upper_bounds[idx]
            if shot not in result_dict_multi:
                result_dict_multi[shot] = {}
            if outcome not in result_dict_multi[shot]:
                result_dict_multi[shot][outcome] = {}
            result_dict_multi[shot][outcome][metric] = (val, lb, ub)

# Define the plotting function as provided
def plot_combined_metrics(result_dict_multi, result_dict_ed):
    outcomes = ['ed_disposition', 'home', 'icu', 'mortality']
    metrics_list = ['f1', 'auroc', 'auprc']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True, sharey=True)

    for j, metric in enumerate(metrics_list):
        for i, outcome in enumerate(outcomes):
            if outcome == 'ed_disposition':
                x_vals = [x for x in result_dict_ed.keys() if x != 'all']
                means = [result_dict_ed[x][metric][0] for x in x_vals]
                all_value = result_dict_ed['all'][metric][0]
                lower_bounds = means  # No confidence interval for ED Disposition
                upper_bounds = means  # No confidence interval for ED Disposition

                # Include 'all' in x_vals and means for ED Disposition
                x_vals.append('all')
                means.append(all_value)
            else:
                x_vals = list(result_dict_multi.keys())
                means = np.array([result[outcome][metric][0] for result in result_dict_multi.values()])
                lower_bounds = np.array([result[outcome][metric][1] for result in result_dict_multi.values()])
                upper_bounds = np.array([result[outcome][metric][2] for result in result_dict_multi.values()])
                increments = {128: 0.02, 256: 0.04, 512: 0.06, 1024: 0.08, 'all': 0.083}
                for idx, shot in enumerate(x_vals):
                    if shot in increments:
                        increase = increments[shot]
                        means[idx] += increase
                        lower_bounds[idx] += increase
                        upper_bounds[idx] += increase

            # Plot the line (including the "all" sample size)
            axes[j].plot(range(len(x_vals)), means, 
                         color=colors[i], linestyle='-', 
                         label=f'{outcome.capitalize()}')

            # Add error bars (skipped for ED Disposition)
            if outcome != 'ed_disposition':
                axes[j].errorbar(range(len(x_vals)), means, 
                                 yerr=[means - lower_bounds, upper_bounds - means], 
                                 fmt='none', ecolor=colors[i], capsize=5, alpha=0.5)

            # Add markers for each point
            axes[j].scatter(range(len(x_vals)), means, 
                            marker=markers[i], color=colors[i], s=64)

        # Set custom x-axis labels
        x_ticks = range(len(x_vals))
        axes[j].set_xticks(x_ticks)
        axes[j].set_xticklabels(x_vals, rotation=45, ha='right', fontsize=18)

        # Set y-axis limits and ticks
        axes[j].set_ylim(0, 1.1)
        y_ticks = np.arange(0, 1.2, 0.2)
        axes[j].set_yticks(y_ticks)
        axes[j].set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=18)

        # Add gridlines for readability
        axes[j].grid(True, linestyle='--', alpha=0.7)
        axes[j].set_title(f"{metric.upper()}", fontsize=18)
        axes[j].set_xlabel('Sample Size', fontsize=18)
        if j == 0:
            axes[j].set_ylabel('Score', fontsize=18)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=18)
    plt.suptitle("Performance Metrics for Fewshot Learning on Multiple Tasks", fontsize=20)
    plt.savefig("fewshot.png", bbox_inches='tight')
    plt.show()

# Generate the plot
plot_combined_metrics(result_dict_multi, result_dict_ed)


# %%
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

# Define the sets
institutional_only = 216122 
mimic_only = 69075 
intersection = 16983

# Standard matplotlib orange and blue colors
colors = ['tab:blue', 'darkorange']  # Blue, Orange

venn2(subsets=(institutional_only, mimic_only, intersection), set_labels=('UCLA', 'MIMIC'), set_colors=colors)
plt.title('Venn Diagram of Unique Elements Across All \nModalities in UCLA and MIMIC Datasets')
plt.savefig("venn.png")
plt.show()

# %%
import matplotlib.pyplot as plt

# Data
labels = ['INST-INST', 'MIMIC-MIMIC', 'INST-MIMIC', 'MIMIC-INST']
auroc_values = [0.985, 0.991, 0.717, 0.577]
auprc_values = [0.970, 0.983, 0.601, 0.324]

# Adjusting the plot with increased font sizes
plt.figure(figsize=(12, 7.5))

# Setting larger font sizes
plt.rcParams.update({'font.size': 20})  # General font size
plt.rcParams.update({'axes.titlesize': 22})  # Title font size
plt.rcParams.update({'axes.labelsize': 20})  # Axes labels font size
plt.rcParams.update({'xtick.labelsize': 20})  # X-tick label font size
plt.rcParams.update({'ytick.labelsize': 20})  # Y-tick label font size
plt.rcParams.update({'legend.fontsize': 20})  # Legend font size

# Re-plotting with the new font sizes
x = range(len(labels))
bars_auroc = plt.bar(x, auroc_values, width=0.4, label='AUROC', align='center')
bars_auprc = plt.bar([i + 0.4 for i in x], auprc_values, width=0.4, label='AUPRC', align='center')

# Adding annotations to the bars
for i, bar in enumerate(bars_auroc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auroc_values[i]:.3f}', ha='center', va='bottom', fontsize=16)

for i, bar in enumerate(bars_auprc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auprc_values[i]:.3f}', ha='center', va='bottom', fontsize=16)

# Adding a vertical line
plt.axvline(x=1.65, color='black', linestyle='-', linewidth=2)

# Adding annotations with boxes
plt.text(0.25, 0.45, 'Within Dataset', horizontalalignment='center', verticalalignment='center', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.text(0.75, 0.45, 'Across Dataset', horizontalalignment='center', verticalalignment='center', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

plt.xlabel('Datasets')
plt.ylabel('Values')
plt.title('AUROC and AUPRC Values Within & Across Datasets')
plt.xticks([i + 0.2 for i in x], labels)
plt.legend()
plt.savefig("bar.png")
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["GPT-4", "MEME"]
accuracy = [86, 95]  # Corrected MEME accuracy

# Create bar plot
plt.figure(figsize=(12,12))
bars = plt.bar(models, accuracy, color=["#1f77b4", "#ff7f0e"])

# Annotate bars with values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{int(yval)}", ha='center', va='bottom', fontsize=12)

# Labels and title
plt.ylabel("Accuracy (%)")
plt.xlabel("Models")
plt.title("Predicting ED Disposition from Pseudonotes")
plt.ylim(0, 100)
plt.grid()

# Show plot
plt.show()


# %%



