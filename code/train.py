ROOT_PATH = '../'

import numpy as np
import sys
sys.path.append(ROOT_PATH + '/code')
from models import XASGNN, SpectrumHead, XASLightningModule, MLPLightningModule
from data import XASGraphDataset, FeatureDataset, collate_fn

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar     

from datetime import datetime
import os, shutil, pickle, yaml, glob, argparse


def cache_features(gnn, dataloader, save_path="absorber_features.pt", device="cuda"):
    gnn.eval().to(device)
    feats_all, spectra_all = [], []

    with torch.no_grad():
        for g, _, spectra in dataloader:
            g = g.to(device)
            spectra = spectra.to(device)
            feats = gnn(g)  # (B, d)
            feats_all.append(feats.cpu())
            spectra_all.append(spectra.cpu())

    feats_all = torch.cat(feats_all, dim=0)
    spectra_all = torch.cat(spectra_all, dim=0)
    torch.save((feats_all, spectra_all), save_path)
    print(f"Saved cached features to {save_path}")
    

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### ------- 0. Parse CLI args -------- ### 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--nblocks", type=int)
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--threebody_cutoff", type=float)
    args = parser.parse_args()

    with open(ROOT_PATH + "/configs/" + args.config, "r") as f:
        config = yaml.safe_load(f)
    # --- Override fields if specified ---
    if args.nblocks is not None:
        config["gnn"]["nblocks"] = args.nblocks
    if args.cutoff is not None:
        config["gnn"]["cutoff"] = args.cutoff
    if args.threebody_cutoff is not None:
        config["gnn"]["threebody_cutoff"] = args.threebody_cutoff
    
    nblocks, cutoff, threebody_cutoff = config["gnn"]["nblocks"], config["gnn"]["cutoff"], config["gnn"]["threebody_cutoff"]
    
    
    ### ------- 1. Load data -------- ### 
    
    train_structures = pickle.load(open(ROOT_PATH + '/dataset/train_structures.pkl', 'rb'))
    train_spectra = torch.load(ROOT_PATH + "/dataset/train_spectra.pt")
    val_structures = pickle.load(open(ROOT_PATH + '/dataset/val_structures.pkl', 'rb'))
    val_spectra = torch.load(ROOT_PATH + "/dataset/val_spectra.pt")

    train_dataset = XASGraphDataset(train_structures, train_spectra, cutoff=cutoff)
    val_dataset = XASGraphDataset(val_structures, val_spectra, cutoff=cutoff)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], collate_fn=collate_fn)

    ### ------- 2. Set up model -------- ### 

    csv_logger = CSVLogger("lightning_logs", name="gnn-%s-%s-%s"%(config['gnn']['nblocks'],config['gnn']['cutoff'], config['gnn']['threebody_cutoff']))
    gnn_config = dict(
        nblocks = config['gnn']['nblocks'], 
        cutoff = config['gnn']['cutoff'], 
        threebody_cutoff = config['gnn']['threebody_cutoff']
    )
    
    head_config = dict(
        hidden_dims = [64, 64], 
        output_size = config['head']['output_size'], 
        drop_rate = config['head']['drop_rate'], 
    )
    model = XASLightningModule(gnn_config, head_config, learning_rate=config['training']['lr'])

    
    ### ------- 3. Train lightning module -------- ### 

    early_stop_callback = EarlyStopping(
        monitor='val_loss',  
        min_delta=1e-6,      
        patience=20,          
        mode='min'        
    )
    
    trainer = pl.Trainer(max_epochs=config['training']['epochs'], accelerator="gpu", devices=1, 
                         callbacks=[early_stop_callback, 
                                    TQDMProgressBar(refresh_rate=0), 
                                    pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)
                                   ], 
                         log_every_n_steps=0,          # disables step-level logging
                         logger=csv_logger,                     
                        )
    print("Train GNN model ... ")
    trainer.fit(model, train_loader, val_loader)


    ### ------- 4. Featurize using GNN model -------- ### 

    loadpath = './lightning_logs/gnn-%s-%s-%s/version_0/'%(nblocks, cutoff, threebody_cutoff)
    ckptfile = glob.glob(loadpath+'checkpoints/*ckpt')[0]
    model = XASLightningModule.load_from_checkpoint(ckptfile)

    features_file = "absorber_features.pt"
    cache_features(model.gnn, train_loader, save_path=loadpath+"/train_" + features_file, device=device)
    cache_features(model.gnn, val_loader, save_path=loadpath+"/val_" + features_file, device=device)

    train_feats, train_spectra = torch.load(loadpath+"/train_" + features_file)
    val_feats, val_spectra = torch.load(loadpath+"/val_" + features_file)
    
    train_dataset_feat = FeatureDataset(train_feats, train_spectra)
    val_dataset_feat = FeatureDataset(val_feats, val_spectra)
    
    train_loader_feat = DataLoader(train_dataset_feat, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader_feat = DataLoader(val_dataset_feat, batch_size=config['training']['batch_size'])


    ### ------- 5. Set up MLP model -------- ### 

    csv_logger = CSVLogger("lightning_logs", name="mlp-%s-%s-%s"%(config['gnn']['nblocks'],config['gnn']['cutoff'], config['gnn']['threebody_cutoff']))
    head_config = dict(
        hidden_dims=config['head']['hidden_dims'], 
        output_size=config['head']['output_size'], 
        drop_rate=config['head']['drop_rate'], 
    )
    mlp_model = MLPLightningModule(head_config, learning_rate=1e-3)

    
    ### ------- 6. Train MLP lightning module -------- ### 

    early_stop_callback = EarlyStopping(
        monitor='val_loss',     # Metric to monitor
        min_delta=1e-6,         # Minimum change to qualify as improvement
        patience=100,             # How many epochs to wait before stopping
        mode='min'              # 'min' because we want to minimize val_loss
    )
    
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        callbacks=[early_stop_callback, 
                TQDMProgressBar(refresh_rate=0), 
                pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)
               ], 
        log_every_n_steps=0,
        logger=csv_logger, 
    )
    
    print("Train MLP model ... ")
    trainer.fit(mlp_model, train_loader_feat, val_loader_feat)



    
