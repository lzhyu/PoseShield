import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from poseshield.pose.dataset import PosesDataset
from poseshield.common.utils import load_model
import copy
from poseshield.pose.losses import poseshield_loss, consistency_score_strict
from poseshield.common.config import get_cfg_defaults, logging_config
import logging

def train_eikonal(
    cfg,
    model,
    train_dataloader,
    test_dataloader,
    logger,
    model_save_dir,
    device,
    num_epochs=20,
    lr=1e-3,
    val_interval=5,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    best_score = float('-inf')   
    best_model_wts = None
    best_model_path = None
    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0.0
        num_batches = 0
        epoch_score = 0.0

        # ------------ Training pass ------------
        for x_batch, y_batch, _ in train_dataloader:
            x_batch = x_batch.to(device).float().reshape(-1, 21*6)
            y_batch = y_batch.to(device).long()
            optimizer.zero_grad()
            loss = poseshield_loss(
                model, 
                x_batch, 
                y_batch,
                dt=cfg.TRAIN.DT,
                grad_loss_weight=cfg.TRAIN.GRAD_LOSS_WEIGHT,
                td_loss_weight=cfg.TRAIN.TD_LOSS_WEIGHT
            )
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

            g = model(x_batch)
            score = consistency_score_strict(g, y_batch)
            epoch_score += score

        avg_loss = epoch_loss / num_batches
        avg_score = epoch_score / num_batches
        logger.info(f"--- Epoch {epoch+1}/{num_epochs} finished, Train Loss = {avg_loss:.4f}, consistency score = {avg_score} ---")
        scheduler.step()
        logger.info(f"Epoch {epoch+1} LR = {scheduler.get_last_lr()[0]:.6e}")
        
        # ------------ Validation pass ------------
        if (epoch + 1) % val_interval == 0 or epoch == 0:
            model.eval()  
            test_loss = 0.0
            test_score = 0.0
            test_batches = 0

            with torch.no_grad():
                for x_test, y_test, _ in test_dataloader:
                    x_test = x_test.to(device).float().reshape(-1, 21*6)
                    y_test = y_test.to(device).long()
                    with torch.enable_grad():
                        val_loss = poseshield_loss(
                            model, 
                            x_test, 
                            y_test,
                            dt=cfg.TRAIN.DT,
                            grad_loss_weight=cfg.TRAIN.GRAD_LOSS_WEIGHT,
                            td_loss_weight=cfg.TRAIN.TD_LOSS_WEIGHT
                        )
                    test_loss += val_loss.item()

                    # consistency score
                    g_test = model(x_test)
                    score_test = consistency_score_strict(g_test, y_test)
                    test_score += score_test
                    test_batches += 1

            avg_test_loss = test_loss / test_batches
            avg_test_score = test_score / test_batches
            logger.warning(f"[Validation @ Epoch {epoch+1}] "
                  f"Loss = {avg_test_loss:.4f}, Score = {avg_test_score:.4f}")

            if avg_test_score > best_score:
                best_score = avg_test_score
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model_path = os.path.join(model_save_dir, f"best_epoch_{epoch+1}_score_{best_score:.4f}.pth")
                torch.save(model.state_dict(), best_model_path)
                logger.warning(f"New best model found at epoch {epoch+1}! Score = {best_score:.4f}")

    return best_model_wts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PoseShield Collision Field Model")
    parser.add_argument("--config-path", default="config_files/basic_config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset using train and test splits directly
    train_dataset = PosesDataset(
        directory_path=cfg.DATA.DIR, 
        split="train"
    )
    test_dataset = PosesDataset(
        directory_path=cfg.DATA.DIR, 
        split="test"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

    exp_dir = os.path.join(".", "experiments", cfg.TRAIN.EXP_NAME)
    os.makedirs(exp_dir, exist_ok=True)
    logger = logging_config(os.path.join(exp_dir, "train.log"))
    model_save_dir = os.path.join(exp_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)

    cfg_save_path = os.path.join(exp_dir, "config.yaml")
    with open(cfg_save_path, 'w') as f:
        f.write(cfg.dump())
    logger.info(f"Config saved to {cfg_save_path}")
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Testing dataset size: {len(test_dataset)}")

    model = load_model(cfg, model_path=None, device=device)

    best_model_weights = train_eikonal(
        cfg,
        model,
        train_dataloader,
        test_dataloader,
        logger,
        model_save_dir,
        device=device,
        num_epochs=cfg.TRAIN.NUM_EPOCHS,
        lr=cfg.TRAIN.LR,
        val_interval=cfg.TRAIN.VAL_INTERVAL,
    )

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        # Save a canonical best_model.pth for easy loading in other scripts
        torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))
        logger.info("Best model weights loaded and copied to best_model.pth.")
    else:
        logger.error("No improvement found during training. No model saved.")
