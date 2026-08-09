# -*- coding: utf-8 -*-
"""Loop training: AMP, class weighting, early stopping, cosine LR, wandb."""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from src.data import make_loader
from src.metrics import binary_metrics, multiclass_metrics
from src import wb


def set_seed(seed: int, deterministic: bool = True):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _class_weights(y, n_out, device):
    y = np.asarray(y)
    y = y[~np.isnan(y.astype(float))].astype(int)
    classes = np.unique(y)
    if len(classes) < 2:
        return None
    w = compute_class_weight("balanced", classes=classes, y=y)
    if n_out == 1:
        # pos_weight untuk BCEWithLogitsLoss = w[1]/w[0]
        d = dict(zip(classes, w))
        return torch.tensor([d.get(1, 1.0) / max(d.get(0, 1.0), 1e-8)],
                            dtype=torch.float32, device=device)
    full = np.ones(n_out, dtype=np.float32)
    for c, ww in zip(classes, w):
        full[int(c)] = ww
    return torch.tensor(full, dtype=torch.float32, device=device)


def train_model(model, train_df, val_df, cfg, label_col, n_out,
                train_tf, eval_tf, seed=42, run=None, tag=""):
    """Latih satu model. Return (model_terbaik, history)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tr_loader = make_loader(train_df, label_col, train_tf, cfg, shuffle=True)
    va_loader = make_loader(val_df, label_col, eval_tf, cfg, shuffle=False)

    cw = _class_weights(train_df[label_col].values.astype(float), n_out, device)
    if n_out == 1:
        crit = nn.BCEWithLogitsLoss(pos_weight=cw)
    else:
        crit = nn.CrossEntropyLoss(weight=cw)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device == "cuda"))

    hist = {"train_loss": [], "val_loss": [], "val_auc": [], "val_bacc": [], "lr": []}
    best_loss, best_state, best_ep, patience = float("inf"), None, -1, 0
    t0 = time.time()

    for ep in range(cfg.epochs):
        # ---------------- train ----------------
        model.train()
        tot, nb = 0.0, 0
        for x, y, _ in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device == "cuda")):
                out = model(x)
                loss = (crit(out.squeeze(-1), y.float()) if n_out == 1
                        else crit(out, y.long()))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss.item()); nb += 1
        tr_loss = tot / max(nb, 1)

        # ---------------- validate ----------------
        model.eval()
        vtot, vnb, ps, ys = 0.0, 0, [], []
        with torch.no_grad():
            for x, y, _ in va_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device == "cuda")):
                    out = model(x)
                    loss = (crit(out.squeeze(-1), y.float()) if n_out == 1
                            else crit(out, y.long()))
                vtot += float(loss.item()); vnb += 1
                p = (torch.sigmoid(out.float()).squeeze(-1) if n_out == 1
                     else torch.softmax(out.float(), 1))
                ps.append(p.cpu().numpy()); ys.append(y.cpu().numpy())
        va_loss = vtot / max(vnb, 1)
        P = np.concatenate(ps); Y = np.concatenate(ys)
        if n_out == 1:
            m = binary_metrics(Y, P)
            auc, bacc = m["AUC"], m["BalancedAcc"]
        else:
            m = multiclass_metrics(Y, P.argmax(1), P, n_classes=n_out)
            auc, bacc = m.get("AUC_ovr", float("nan")), m["BalancedAcc"]

        sched.step()
        lr_now = sched.get_last_lr()[0]
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["val_auc"].append(float(auc)); hist["val_bacc"].append(float(bacc))
        hist["lr"].append(float(lr_now))

        print(f"    ep {ep+1:02d}/{cfg.epochs}  train {tr_loss:.4f}  "
              f"val {va_loss:.4f}  AUC {auc:.4f}  bACC {bacc:.4f}")
        wb.log(run, {f"{tag}/train_loss": tr_loss, f"{tag}/val_loss": va_loss,
                     f"{tag}/val_auc": auc, f"{tag}/val_bacc": bacc,
                     f"{tag}/lr": lr_now, f"{tag}/epoch": ep + 1})

        if va_loss < best_loss - 1e-5:
            best_loss, best_ep, patience = va_loss, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"    early stop @ epoch {ep+1} (best {best_ep+1})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_ep + 1
    hist["best_val_loss"] = best_loss
    hist["train_minutes"] = (time.time() - t0) / 60.0
    hist["gpu_peak_mb"] = (torch.cuda.max_memory_allocated() / 1024**2
                           if device == "cuda" else 0.0)
    print(f"    selesai {hist['train_minutes']:.1f} menit | "
          f"VRAM puncak {hist['gpu_peak_mb']:.0f} MB")
    return model, hist
