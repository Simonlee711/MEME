# %%
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np 
# %%
merged_txt_df = pd.read_csv('Finetune-merged-txt.csv')
fusion_txt_df = pd.read_csv('Finetune-fusion-v2-epoch0.csv')
​
result_dict = {
    "Single": merged_txt_df,
    "Multi": fusion_txt_df
}
​
# %%
def bootstrap_scores(yt, yp, n=1000):
    aurocs, auprcs, f1s = [], [], []
    preds = (yp > 0.5).astype(int)
    for i in range(n):
        idx = np.random.choice(len(yt), len(yt), replace=True)
        aurocs.append(roc_auc_score(yt[idx], yp[idx]))
        auprcs.append(average_precision_score(yt[idx], yp[idx]))
        f1s.append(f1_score(yt[idx], preds[idx]))
    return {
        'auroc': (np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)),
        'auprc': (np.percentile(auprcs, 2.5), np.percentile(auprcs, 97.5)),
        'f1': (np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))
    }
​
# %%
​
​
f, axarr = plt.subplots(1, 2, figsize=(11, 4))
for model_name, df in result_dict.items():
    # model_name = "Merged-txt"
    yt = df['y_true']
    yp = df['y_prob']
    fpr, tpr, _ = roc_curve(yt, yp)
    precisions, recalls, _ = precision_recall_curve(yt, yp)
    auroc = roc_auc_score(yt, yp)
    auprc = average_precision_score(yt, yp)
    preds = (yp >= 0.5).astype(int)
    f1 = f1_score(yt, preds)
​
    bootstraps = bootstrap_scores(yt, yp, n=1000)
​
    print(f"{model_name} f1: {f1:.3f} ({bootstraps['f1'][0]:.3f}, {bootstraps['f1'][1]:.3f})")
    axarr[0].plot(fpr, tpr, alpha=0.6, label=f'{model_name}: {auroc:.3f}({bootstraps["auroc"][0]:.3f}, {bootstraps["auroc"][1]:.3f})')
    axarr[1].step(recalls, precisions, where='post', alpha=0.6, label=f'{model_name}: {auprc:.3f}({bootstraps["auprc"][0]:.3f}, {bootstraps["auprc"][1]:.3f})')
​
​
axarr[0].legend()
axarr[0].set_xlabel('FPR (1 - Specificity)')
axarr[0].set_ylabel('TPR (Sensitivity)')
axarr[1].legend()
axarr[1].set_xlabel('Recall (Sensitivity)')
axarr[1].set_ylabel('Precision (PPV)')
# %%
