from __future__ import annotations
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union, List, Dict
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset

HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c:i for i,c in enumerate(HAM10000_CLASSES)}
IDX_TO_CLASS = {i:c for c,i in CLASS_TO_IDX.items()}

def load_state_dict_flexible(checkpoint_path, map_location="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        obj = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(obj, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], Mapping):
                obj = obj[key]
                break
    if any(str(k).startswith("module.") for k in obj.keys()):
        obj = {str(k).replace("module.", "", 1):v for k,v in obj.items()}
    return obj

def load_model_weights(model, checkpoint_path, device, strict=True):
    state = load_state_dict_flexible(checkpoint_path, device)
    model.load_state_dict(state, strict=strict)
    return model.to(device).eval()

def resolve_image_path(row: pd.Series, data_root, search_dirs=None):
    data_root = Path(data_root)
    for col in ("image_path","filepath","file_path","path","image","filename","file_name"):
        if col in row.index and pd.notna(row[col]):
            raw=Path(str(row[col]))
            for p in (raw, data_root/raw, data_root.parent/raw):
                if p.exists(): return p.resolve()
    image_id=str(row["image_id"])
    dirs=[data_root,data_root/"raw",data_root/"HAM10000_images_part_1",data_root/"HAM10000_images_part_2",data_root/"raw"/"HAM10000_images_part_1",data_root/"raw"/"HAM10000_images_part_2"]
    if search_dirs: dirs += [Path(x) for x in search_dirs]
    for d in dirs:
        for ext in (".jpg",".jpeg",".png"):
            p=d/f"{image_id}{ext}"
            if p.exists(): return p.resolve()
    raise FileNotFoundError(f"Cannot find image for image_id={image_id} under {data_root}")

class ErrorAnalysisDataset(Dataset):
    def __init__(self, dataframe, transform, data_root, class_to_idx=None):
        self.df=dataframe.reset_index(drop=True).copy(); self.transform=transform; self.data_root=Path(data_root); self.class_to_idx=dict(class_to_idx or CLASS_TO_IDX)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]; path=resolve_image_path(row,self.data_root)
        image=Image.open(path).convert("RGB")
        if self.transform: image=self.transform(image)
        return image, torch.tensor(self.class_to_idx[str(row["dx"])],dtype=torch.long), idx

@torch.inference_mode()
def collect_predictions(model, data_loader, device):
    model.eval(); yt=[]; yp=[]; probs=[]; idxs=[]
    for images, labels, indices in data_loader:
        images=images.to(device); labels=labels.to(device).long(); logits=model(images); p=torch.softmax(logits,dim=1); pred=p.argmax(dim=1)
        yt.extend(labels.cpu().numpy()); yp.extend(pred.cpu().numpy()); probs.extend(p.cpu().numpy()); idxs.extend(indices.cpu().numpy())
    probs=np.asarray(probs,dtype=np.float32)
    return {"y_true":np.asarray(yt),"y_pred":np.asarray(yp),"y_prob":probs,"confidence":probs.max(axis=1),"sample_idx":np.asarray(idxs)}

def build_prediction_dataframe(test_df, prediction_output, idx_to_class=None):
    idx_to_class=dict(idx_to_class or IDX_TO_CLASS); out=test_df.iloc[prediction_output["sample_idx"]].reset_index(drop=True).copy()
    out["true_idx"]=prediction_output["y_true"]; out["pred_idx"]=prediction_output["y_pred"]
    out["true_class"]=out["true_idx"].map(idx_to_class); out["pred_class"]=out["pred_idx"].map(idx_to_class)
    out["confidence"]=prediction_output["confidence"]; out["is_correct"]=out["true_idx"]==out["pred_idx"]
    out["error_pair"]=out["true_class"].astype(str)+" → "+out["pred_class"].astype(str)
    for i,c in idx_to_class.items(): out[f"prob_{c}"]=prediction_output["y_prob"][:,int(i)]
    return out

def class_error_summary(df, class_names=None):
    rows=[]
    for c in list(class_names or HAM10000_CLASSES):
        s=df[df.true_class==c]; n=len(s); e=int((~s.is_correct).sum()) if n else 0
        rows.append({"class":c,"n_samples":n,"n_correct":n-e,"n_errors":e,"accuracy":(n-e)/n if n else np.nan,"error_rate":e/n if n else np.nan,"mean_confidence":s.confidence.mean() if n else np.nan,"mean_confidence_on_errors":s.loc[~s.is_correct,"confidence"].mean() if e else np.nan})
    return pd.DataFrame(rows)

def common_confusion_pairs(df, top_n=10):
    e=df[~df.is_correct]
    if e.empty: return pd.DataFrame(columns=["true_class","pred_class","count","mean_confidence","max_confidence"])
    return e.groupby(["true_class","pred_class"],observed=True).agg(count=("is_correct","size"),mean_confidence=("confidence","mean"),max_confidence=("confidence","max")).reset_index().sort_values(["count","mean_confidence"],ascending=[False,False]).head(top_n).reset_index(drop=True)

def high_confidence_errors(df, threshold=.80, top_n=None):
    x=df[(~df.is_correct)&(df.confidence>=threshold)].sort_values("confidence",ascending=False).reset_index(drop=True)
    return x.head(top_n) if top_n else x

def low_confidence_predictions(df, threshold=.50, top_n=None):
    x=df[df.confidence<threshold].sort_values("confidence").reset_index(drop=True)
    return x.head(top_n) if top_n else x

def confusion_count_matrix(df, class_names=None):
    n=len(class_names or HAM10000_CLASSES); return confusion_matrix(df.true_idx,df.pred_idx,labels=list(range(n)))
def normalized_confusion_matrix(df, class_names=None):
    n=len(class_names or HAM10000_CLASSES); return confusion_matrix(df.true_idx,df.pred_idx,labels=list(range(n)),normalize="true")
def error_overview(df):
    n=len(df); e=int((~df.is_correct).sum()); return {"n_samples":n,"n_correct":n-e,"n_errors":e,"accuracy":float(df.is_correct.mean()),"error_rate":float((~df.is_correct).mean()),"mean_confidence":float(df.confidence.mean()),"mean_error_confidence":float(df.loc[~df.is_correct,"confidence"].mean()) if e else np.nan}
def save_error_analysis_tables(df, output_dir, model_name, high_conf_threshold=.80):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); s=model_name.lower().replace(" ","_").replace("-","_")
    paths={"predictions":output_dir/f"{s}_predictions.csv","class_errors":output_dir/f"{s}_class_errors.csv","confusion_pairs":output_dir/f"{s}_confusion_pairs.csv","high_confidence_errors":output_dir/f"{s}_high_confidence_errors.csv"}
    df.to_csv(paths["predictions"],index=False); class_error_summary(df).to_csv(paths["class_errors"],index=False); common_confusion_pairs(df,20).to_csv(paths["confusion_pairs"],index=False); high_confidence_errors(df,high_conf_threshold).to_csv(paths["high_confidence_errors"],index=False)
    return paths
