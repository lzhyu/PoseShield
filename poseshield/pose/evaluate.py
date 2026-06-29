import os

import torch
from torch.utils.data import DataLoader
from poseshield.pose.dataset import PosesDataset
from poseshield.common.utils import load_model
from poseshield.pose.losses import consistency_score_strict, poseshield_loss
from poseshield.common.config import get_cfg_defaults, logging_config
def evaluate(
    cfg,
    test_dataset,
    model_path,
    batch_size=32,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = load_model(cfg, model_path, device)

    # logging
    exp_dir = os.path.join(".", "experiments", cfg.TRAIN.EXP_NAME)
    os.makedirs(exp_dir, exist_ok=True)
    logger = logging_config(os.path.join(exp_dir, "eval.log"))

    logger.info(f"Evaluating model loaded from: {model_path}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    total_loss = 0.0
    num_batches = 0

    # Confusion matrix counts (positive = collision, y == -1)
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for x_test, y_test, _ in test_loader:
            x_test = x_test.to(device).float().reshape(-1, 21*6)
            y_test = y_test.to(device).long().view(-1)

            # loss
            with torch.enable_grad():
                loss = poseshield_loss(
                    model, x_test, y_test, 
                    dt=cfg.TRAIN.DT, 
                    grad_loss_weight=cfg.TRAIN.GRAD_LOSS_WEIGHT, 
                    td_loss_weight=cfg.TRAIN.TD_LOSS_WEIGHT
                )
            total_loss += loss.item()
            num_batches += 1

            # predictions
            g_test = model(x_test).view(-1)

            pred_collision = g_test < 0
            pred_no_collision = ~pred_collision

            gt_collision = y_test == -1
            gt_no_collision = y_test == 1

            tp += (pred_collision & gt_collision).sum().item()
            tn += (pred_no_collision & gt_no_collision).sum().item()
            fp += (pred_collision & gt_no_collision).sum().item()
            fn += (pred_no_collision & gt_collision).sum().item()

    total_samples = tp + tn + fp + fn
    avg_loss = total_loss / max(num_batches, 1)
    avg_score = (tp + tn) / total_samples if total_samples > 0 else 0.0

    # overall confusion matrix
    logger.info("[Evaluation] Confusion Matrix (collision = positive)")
    if total_samples > 0:
        tp_rate = tp / total_samples
        fp_rate = fp / total_samples
        fn_rate = fn / total_samples
        tn_rate = tn / total_samples
    else:
        tp_rate = fp_rate = fn_rate = tn_rate = 0.0
    logger.info(
        f"TP: {tp} (rate={tp_rate:.4f}), FP: {fp} (rate={fp_rate:.4f}), "
        f"FN: {fn} (rate={fn_rate:.4f}), TN: {tn} (rate={tn_rate:.4f})"
    )
    logger.info(f"[Evaluation] Accuracy = {avg_score:.4f}")

    logger.info(f"[Evaluation] Loss = {avg_loss:.4f}")


    return avg_loss, avg_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PoseShield Collision Field Model")
    parser.add_argument("--config-path", default="config_files/basic_config.yaml", help="Path to config file")
    parser.add_argument("--model-path", default=None, help="Path to specific model weights (if None, uses best model from exp_dir)")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg.freeze()

    # Keep test set consistent with training
    test_ds = PosesDataset(
        directory_path=cfg.DATA.DIR, 
        split="test"
    )

    best_model_path = args.model_path
    if best_model_path is None:
        exp_dir = os.path.join(".", "experiments", cfg.TRAIN.EXP_NAME)
        model_save_dir = os.path.join(exp_dir, "model")
        model_files = [f for f in os.listdir(model_save_dir) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No model found in {model_save_dir}")
        # Sort by epoch and take the last one, or prioritize best_model.pth
        if "best_model.pth" in model_files:
            best_model_path = os.path.join(model_save_dir, "best_model.pth")
        else:
            model_files = sorted(
                model_files,
                key=lambda x: int(x.split('_')[2]) if (len(x.split('_')) > 2 and x.split('_')[2].isdigit()) else 0
            )
            best_model_path = os.path.join(model_save_dir, model_files[-1])
    
    print(f"Using model for evaluation: {best_model_path}")

    # Run evaluation
    evaluate(
        cfg=cfg,
        test_dataset=test_ds,
        model_path=best_model_path,
        batch_size=cfg.TRAIN.BATCH_SIZE,
    )
