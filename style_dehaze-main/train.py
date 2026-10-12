import argparse
import csv
import json
import os

import cv2
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.utils.utils import get_config
from model.trainers.trainer import Trainer
from dataset.dataset_ImagePair import ImagePairDataset
from logger import TrainingLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default='./config/Dataset_OHAZE.yaml')
    parser.add_argument('--root_dir', default='./dataset/Dataset_OHAZE')
    parser.add_argument('--job_name', default=None)
    parser.add_argument('--wandb_project', default='TESIS_CLEAN')
    parser.add_argument('--no_wandb', action='store_true')
    return parser.parse_args()


def has_split(root, phase):
    return (
        os.path.isdir(os.path.join(root, f'{phase}A')) and
        os.path.isdir(os.path.join(root, f'{phase}B'))
    )


def append_metrics_csv(path, row):
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def write_metrics_summary(path, summary):
    with open(path, 'w') as f:
        json.dump(to_jsonable(summary), f, indent=2)


def mean_losses(loss_rows):
    arr = np.array(loss_rows, dtype=np.float64)
    return {
        'loss_total': float(arr[:, 0].mean()),
        'loss_content': float(arr[:, 1].mean()),
        'loss_style': float(arr[:, 2].mean()),
        'loss_recon': float(arr[:, 3].mean()),
        'loss_pixel': float(arr[:, 4].mean()),
        'loss_smooth': float(arr[:, 5].mean()),
        'loss_scm': float(arr[:, 6].mean()),
    }


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def update_training_plots(log_path, train_metrics_path, validation_metrics_path, test_metrics_path):
    plot_dir = os.path.join(log_path, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    train_rows = read_csv_rows(train_metrics_path)
    if train_rows:
        epochs = [int(row['epoch']) for row in train_rows]
        plt.figure(figsize=(9, 5))
        for col in ['loss_total', 'loss_pixel', 'loss_content', 'loss_recon', 'loss_smooth']:
            if col in train_rows[0]:
                vals = [float(row[col]) for row in train_rows]
                plt.plot(epochs, vals, marker='o', label=col)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'train_losses.png'), dpi=180)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, [float(row['lr']) for row in train_rows], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'learning_rate.png'), dpi=180)
        plt.close()

    metric_rows = []
    for path in [validation_metrics_path, test_metrics_path]:
        metric_rows.extend(read_csv_rows(path))
    if metric_rows:
        for metric in ['psnr', 'ssim']:
            plt.figure(figsize=(8, 4.5))
            splits = sorted(set(row['split'] for row in metric_rows))
            for split in splits:
                rows = [row for row in metric_rows if row['split'] == split]
                plt.plot(
                    [int(row['epoch']) for row in rows],
                    [float(row[metric]) for row in rows],
                    marker='o',
                    label=split,
                )
            plt.xlabel('Epoch')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} Curves')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{metric}_curves.png'), dpi=180)
            plt.close()


def evaluate_loader(trainer, loader, epoch, split_name, metrics_path, summary_path):
    print(f'\n\nValidating on {split_name} set ......')
    avg_psnr = []
    avg_ssim = []
    eval_bar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f'{split_name} Epoch {epoch}',
    )

    for batch_id, (source_image, gt_image) in eval_bar:
        psnr, ssim = trainer.test(
            epoch,
            batch_id,
            source_image,
            gt_image,
            len_data=len(loader),
            suffix=split_name,
        )
        avg_psnr.append(psnr)
        avg_ssim.append(ssim)

    mean_psnr = float(sum(avg_psnr) / len(avg_psnr))
    mean_ssim = float(sum(avg_ssim) / len(avg_ssim))
    row = {
        'epoch': epoch,
        'split': split_name,
        'psnr': mean_psnr,
        'ssim': mean_ssim,
    }
    append_metrics_csv(metrics_path, row)
    write_metrics_summary(summary_path, row)
    print(f'\nEpoch: {epoch} - {split_name.upper()} Avg PSNR: {mean_psnr:.4f}, Avg SSIM: {mean_ssim:.4f} ')
    return mean_psnr, mean_ssim


def main():
    cli_args = parse_args()
    cfg_path = cli_args.cfg_path
    root_dir = cli_args.root_dir
    wandb_project = cli_args.wandb_project

    args = get_config(cfg_path)
    args['cfg_path'] = cfg_path
    if cli_args.job_name is not None:
        args['job_name'] = cli_args.job_name
    if cli_args.no_wandb:
        args['wandb'] = None

    code_name = args.get('job_name', os.path.splitext(os.path.basename(cfg_path))[0])
    resume = args['resume']
    img_size = (args['img_h'], args['img_w'])
    batch_size = args['batch_size']
    wandb_key = args.get('wandb')
    val_freq = args.get('val_freq', 10)
    test_freq = args.get('test_freq', 10)

    if wandb_key is not None:
        if wandb is None:
            raise ImportError('wandb is not installed. Install wandb or run with --no_wandb.')
        print('\n\nSetting up wandb\n\n')
        wandb.login(key=wandb_key)
        wandb_run = wandb.init(
            project=wandb_project,
            entity='alfin-nurhalim',
            name=code_name,
            config=args,
        )
    else:
        wandb_run = None

    train_dataset = ImagePairDataset(
        root_dir=root_dir,
        phase='train',
        augment=args['aug'],
        stage=args['stage'],
        img_size=img_size,
    )
    test_dataset = ImagePairDataset(
        root_dir=root_dir,
        phase='test',
        augment=False,
        stage=args['stage'],
        img_size=img_size,
    )
    val_dataset = None
    if has_split(root_dir, 'val'):
        val_dataset = ImagePairDataset(
            root_dir=root_dir,
            phase='val',
            augment=False,
            stage=args['stage'],
            img_size=img_size,
        )

    print(f'\n\nTrain Dataset : {len(train_dataset)}')
    if val_dataset is not None:
        print(f'Val Dataset : {len(val_dataset)}')
    print(f'Test Dataset : {len(test_dataset)}\n\n')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print('attention:', args['attention'])
    print('FLOW :', args['n_flow'], 'BLOCK: ', args['n_block'])
    print('epoch: ', args['total_epoch'])
    print('\n\n')

    trainer = Trainer(args)
    if resume is not None:
        trainer.load_model(resume, flow_only=False)

    training_logger = TrainingLogger(trainer.log_path)
    validation_metrics_path = os.path.join(trainer.log_path, 'validation_metrics.csv')
    validation_summary_path = os.path.join(trainer.log_path, 'validation_latest.json')
    test_metrics_path = os.path.join(trainer.log_path, 'test_metrics.csv')
    test_summary_path = os.path.join(trainer.log_path, 'test_latest.json')
    train_metrics_path = os.path.join(trainer.log_path, 'training_metrics.csv')
    best_metrics_path = os.path.join(trainer.log_path, 'best_metrics.json')
    best_val_psnr = float('-inf')
    best_val_ssim = float('-inf')
    best_test_psnr = float('-inf')
    best_test_ssim = float('-inf')
    best_metrics = {}

    for epoch in range(args['total_epoch']):
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Epoch {epoch}',
        )
        last_loss = None
        epoch_loss_rows = []

        for batch_id, (source_image, gt_image) in progress_bar:
            loss_list = trainer.train(batch_id, source_image, gt_image, epoch)
            loss, loss_c, loss_s, loss_r, loss_p, loss_smooth, loss_scm = (
                loss_list if loss_list is not None else [0, 0, 0, 0, 0, 0, 0]
            )
            last_loss = loss
            epoch_loss_rows.append([loss, loss_c, loss_s, loss_r, loss_p, loss_smooth, loss_scm])

            if loss_list is not None and wandb_key is not None:
                wandb.log({
                    'train/loss_total': loss,
                    'train/loss_content': loss_c,
                    'train/loss_style': loss_s,
                    'train/loss_restoration': loss_r,
                    'train/loss_pixel': loss_p,
                    'train/loss_smooth': loss_smooth,
                    'train/loss_scm': loss_scm,
                }, step=epoch)

            progress_bar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Loss_c': f'{loss_c:.4f}',
                'Loss_s': f'{loss_s:.4f}',
                'Loss_r': f'{loss_r:.4f}',
                'loss_p': f'{loss_p:.4f}',
                'Loss_smooth': f'{loss_smooth:.4f}',
                'loss_scm': f'{loss_scm:.4f}',
            })

        if last_loss is not None:
            training_logger.log_epoch_loss(epoch, last_loss)
        print(f'\nEpoch: {epoch} - Last Batch Loss: {last_loss:.4f}')

        train_row = {
            'epoch': epoch,
            'lr': trainer.get_lr(),
            'last_loss': last_loss,
        }
        if epoch_loss_rows:
            train_row.update(mean_losses(epoch_loss_rows))
        append_metrics_csv(train_metrics_path, train_row)

        if 'train' in args['eval'] and epoch % test_freq == 0:
            train_psnr, train_ssim = evaluate_loader(
                trainer,
                train_loader,
                epoch,
                'train',
                os.path.join(trainer.log_path, 'train_eval_metrics.csv'),
                os.path.join(trainer.log_path, 'train_eval_latest.json'),
            )
            if wandb_key is not None:
                wandb.log({'train/psnr': train_psnr, 'train/ssim': train_ssim}, step=epoch)

        if val_loader is not None and val_freq > 0 and epoch % val_freq == 0:
            val_psnr, val_ssim = evaluate_loader(
                trainer,
                val_loader,
                epoch,
                'val',
                validation_metrics_path,
                validation_summary_path,
            )
            if wandb_key is not None:
                wandb.log({'val/psnr': val_psnr, 'val/ssim': val_ssim}, step=epoch)
            trainer.save_model(
                f'epoch_{epoch:03d}_val',
                epoch=epoch,
                metrics={'val_psnr': val_psnr, 'val_ssim': val_ssim},
            )
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_metrics['best_val_psnr'] = {'epoch': epoch, 'psnr': val_psnr, 'ssim': val_ssim}
                trainer.save_model('best_val_psnr', epoch=epoch, metrics=best_metrics['best_val_psnr'])
            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                best_metrics['best_val_ssim'] = {'epoch': epoch, 'psnr': val_psnr, 'ssim': val_ssim}
                trainer.save_model('best_val_ssim', epoch=epoch, metrics=best_metrics['best_val_ssim'])
            write_metrics_summary(best_metrics_path, best_metrics)

        if test_freq > 0 and epoch % test_freq == 0:
            test_psnr, test_ssim = evaluate_loader(
                trainer,
                test_loader,
                epoch,
                'test',
                test_metrics_path,
                test_summary_path,
            )
            if wandb_key is not None:
                wandb.log({'test/psnr': test_psnr, 'test/ssim': test_ssim}, step=epoch)
            trainer.save_model(
                f'epoch_{epoch:03d}_test',
                epoch=epoch,
                metrics={'test_psnr': test_psnr, 'test_ssim': test_ssim},
            )
            if test_psnr > best_test_psnr:
                best_test_psnr = test_psnr
                best_metrics['best_test_psnr'] = {'epoch': epoch, 'psnr': test_psnr, 'ssim': test_ssim}
                trainer.save_model('best_test_psnr', epoch=epoch, metrics=best_metrics['best_test_psnr'])
            if test_ssim > best_test_ssim:
                best_test_ssim = test_ssim
                best_metrics['best_test_ssim'] = {'epoch': epoch, 'psnr': test_psnr, 'ssim': test_ssim}
                trainer.save_model('best_test_ssim', epoch=epoch, metrics=best_metrics['best_test_ssim'])
            write_metrics_summary(best_metrics_path, best_metrics)

        trainer.save_model('last', epoch=epoch, metrics=train_row)
        update_training_plots(trainer.log_path, train_metrics_path, validation_metrics_path, test_metrics_path)
        print(f'Learning rate after epoch {epoch}: {trainer.get_lr():.8f}')
        print('\n\n')

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
