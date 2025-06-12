# necessary imports torch, numpy, etc.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# models and data imports
from data import FORAge, FORAge_H5
from model import MAE3D, MAE3D_reg
from util import cal_loss,  IOStream

# metrics
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError

# wandb for metric logging and visualization 
import wandb


# hydra imports 
import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate

# logging 
import logging


def _init_(cfg):
        # setup experiment dirs 
        experiment_dir = Path(f'./experiments/{cfg.experiment_setup.name}')
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        for i in ['checkpoints', 'visualization', 'logs']:
            (experiment_dir / i).mkdir(parents=True, exist_ok=True)


        logger = logging.getLogger(__name__) 

        logger.info(f'Experiment initalized: {cfg.experiment_setup.name}, mode: {cfg.mode}')

        # paths to config 
        cfg.experiment_setup.experiment_dir = experiment_dir
        cfg.experiment_setup.checkpoints_dir = experiment_dir / 'checkpoints'
        cfg.experiment_setup.visualization_dir = experiment_dir / 'visualization'
        cfg.experiment_setup.logs_dir = experiment_dir / 'logs'

        return logger


def _set_reproducability(seed, deterministic=True):
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def training

def train(cfg, logger):
    # setup wandb for loggin metrics 
    wandb.init(
        project=f'MAE3D-SingleTree', 
        name=f'{cfg.experiment_setup.name}_{cfg.mode}',
        mode='online' # online streaming of metrics
    )

    # setup datasets no other datasets are supported yet
    if cfg.downstream.dataset == 'FORAGE':
        train_loader = DataLoader(
            FORAge_H5(split='train', fraction=cfg.downstream.limited_ratio),
            num_workers=cfg.downstream.num_workers, 
            batch_size=cfg.downstream.batch_size, 
            shuffle=cfg.downstream.shuffle, 
            drop_last=cfg.downstream.drop_last,
            pin_memory=cfg.downstream.pin_memory,
            prefetch_factor=cfg.downstream.prefetch_factor
        )

        val_loader = DataLoader(
            FORAge_H5(split='val', fraction=1), # always use full validation set
            num_workers=cfg.downstream.num_workers, 
            batch_size=cfg.downstream.val_batch_size, 
            shuffle=False, # no need to shuffle validation set
            drop_last=False, # no drop of last for validation
            pin_memory=cfg.downstream.pin_memory,
            prefetch_factor=cfg.downstream.prefetch_factor
        )

        test_loader = DataLoader(
            FORAge_H5(split='test', fraction=1), # always use full test set
            num_workers=cfg.downstream.num_workers, 
            batch_size=cfg.downstream.val_batch_size, 
            shuffle=False, 
            drop_last=False,
            pin_memory=cfg.downstream.pin_memory,
            prefetch_factor=cfg.downstream.prefetch_factor
        )

    # log dataset and fraction information 
    logger.info(
        f'Using dataset: {cfg.downstream.dataset}, limited ratio: {cfg.downstream.limited_ratio}')

    # set device
    device = torch.device("cuda" if cfg.experiment_setup.use_cuda else "cpu")




   
    

    val_r2_reps = []
    val_rmse_reps = []
    test_r2_reps = []
    test_rmse_reps = []
    
    # FOR EACH REPETITION:
    for i in range(args.repetitions):

        writer = SummaryWriter(f'checkpoints/{file_name}/tensorboard/{args.limited_ratio}_rep_{i}')

        
        model_pretrain = MAE3D(args, encoder_dims=1024, decoder_dims=1024).to(device)
        model_reg = MAE3D_reg(dropout=args.dropout, encoder_dims=1024).to(device)
        
        io.cprint(str(model_pretrain))
        io.cprint(str(model_reg))

        # CHANGED loading 
        if args.pretrained: 
            print(f'Loading model from: {pretrain_model_path}')
            pretrained_dict = torch.load(pretrain_model_path)

            # extract weights from patch_embed 
            point_embed_dict = {k: v for k, v in pretrained_dict.items() if 'patch_embed' in k}

            model_reg_dict = model_reg.state_dict() 

            for k, v in point_embed_dict.items():
                if k in model_reg_dict and model_reg_dict[k].size() == v.size():
                    model_reg_dict[k] = v

            model_reg.load_state_dict(model_reg_dict, strict=False)

            io.cprint(f'Pretrained model loaded')
        
        else: 
            io.cprint(f'No pretrained model loaded.')


        if args.linear_classifier:
            for n, p in model_reg.named_parameters():
                if n.split('.')[1] == 'patch_embed':
                    p.requires_grad = False
            io.cprint(f'Linear probing mode. Updating only regression head.')
        elif args.finetune: 
            io.cprint(f'Full fine-tuning mode. Updating all parameters (embeding module and regression head).')
        elif args.from_scratch: 
            io.cprint(f'Full training from scratch. Updating all parameters (embedding and regressor head). No pretrained weights loaded.')

        # setup optimizer, schedulr and criterion
        if args.use_sgd:
            print("Use SGD")
            opt = optim.SGD([p for p in model_reg.parameters() if p.requires_grad], lr=args.lr * 100,
                            momentum=args.momentum, weight_decay=5e-4)
        else:
            print("Use Adam")
            opt = optim.Adam([p for p in model_reg.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)


        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
        criterion = nn.SmoothL1Loss()
        
        # set up torchmetrics, bring to device 
        r2_train = R2Score().to(device)
        mse_train = MeanSquaredError().to(device)

        r2_val = R2Score().to(device)
        mse_val = MeanSquaredError().to(device)
        mae_val = MeanAbsoluteError().to(device)
    
    


        rep = i + 1
        io.cprint(f'Running repetition {rep} of {args.repetitions}')#

        # best rmse, r2 to save best model
        best_val_rmse =  1e10 # best model has to be <
        best_val_r2 = -1e10 # best model has to be > 

        # reset loss for each repetition 
        train_loss = 0.0
        val_loss = 0.0
        count = 0.0

        for epoch in range(args.epochs):
            #scheduler.step()
            train_loss = 0.0
            count = 0.0
            model_reg.train()
            total_time = 0.0

            r2_train.reset() 
            mse_train.reset() 

            for batch, (data, labels) in enumerate(tqdm(train_loader)):
                data, labels = data.to(device).float(), labels.squeeze(-1).to(device)
                data = data.permute(0, 2, 1) 
                batch_size = data.size()[0]

                opt.zero_grad()
                start_time = time.time()
                preds = model_reg(data)

                loss = criterion(preds.squeeze(-1), labels)
                loss.backward()
                opt.step()
                end_time = time.time()
                total_time += (end_time - start_time)
                
                count += batch_size
                train_loss += loss.item() * batch_size
                
                r2_train.update(preds.squeeze(-1), labels) 
                mse_train.update(preds.squeeze(-1), labels)


            # scheduler step after each epoch
            scheduler.step()
            print('train total time is', total_time)
            
            epoch_r2 = r2_train.compute().detach().cpu().item()
            epoch_mse = mse_train.compute()
            epoch_rmse = torch.sqrt(epoch_mse).detach().cpu().item()
            

            outstr = 'Train %d, loss: %.6f, train rmse: %.6f, train r2: %.6f' % (
                epoch,
                train_loss * 1.0 / count,
                epoch_rmse, 
                epoch_r2
            )
            io.cprint(outstr)

            writer.add_scalar('train/epoch_loss', train_loss * 1.0 / count, epoch)
            writer.add_scalar('train/epoch_rmse', epoch_rmse, epoch)
            writer.add_scalar('train/epoch_r2', epoch_r2, epoch)

            writer.flush()
            
            mae_val.reset()
            mse_train.reset() 
            r2_train.reset()

            # validate every 10 epochs, save best model every 10 epochs
            if epoch % 10 == 0:

                ####################
                # Val
                ####################
                val_loss = 0.0
                count = 0.0
                total_time = 0.0


                # set model in evaluation mode 
                model_reg.eval() 



                mae_val.reset() 
                mse_val.reset() 
                r2_val.reset()

                # list for savind predictions and gt
                gt = []
                predictions = []

                with torch.no_grad():
                    for data, labels in tqdm(val_loader):
                        data, labels = data.to(device).float(), labels.squeeze(-1).to(device)
                        data = data.permute(0, 2, 1)
                        batch_size = data.size()[0]


                        start_time = time.time()
                        preds = model_reg(data)
                        loss = criterion(preds.squeeze(-1), labels)
                        end_time = time.time()
                        total_time += (end_time - start_time)
                        

                        count += batch_size
                        val_loss += loss.item() * batch_size

                        r2_val.update(preds.squeeze(-1), labels)
                        mse_val.update(preds.squeeze(-1), labels)
                        mae_val.update(preds.squeeze(-1), labels)

                        # append predictions and gt to list 
                        gt.append(labels)
                        predictions.append(preds)
                        

                    print('eval total time is', total_time)


                    epoch_val_r2 = r2_val.compute().detach().cpu().item()
                    epoch_val_mse = mse_val.compute()
                    epoch_val_rmse = torch.sqrt(epoch_val_mse).detach().cpu().item()
                    epoch_val_mae = mae_val.compute().detach().cpu().item()

                    epoch_val_mbe = torch.mean(torch.cat(predictions) - torch.cat(gt)).detach().cpu().item()

                    r2_val.reset()
                    mse_val.reset()
                    mae_val.reset()


                    writer.add_scalar('val/epoch_loss', val_loss * 1.0 / count, epoch)
                    writer.add_scalar('val/epoch_rmse', epoch_val_rmse, epoch)
                    writer.add_scalar('val/epoch_r2', epoch_val_r2, epoch)
                    writer.add_scalar('val/epoch_mae', epoch_val_mae, epoch)
                    writer.add_scalar('val/epoch_mbe', epoch_val_mbe, epoch)
                    writer.add_scalar('val/epoch_mse', epoch_val_mse, epoch)

                    writer.flush()

                    outstr = f'Validation {epoch}, loss: {(val_loss * 1.0 / count):.2}, val rmse: {epoch_val_rmse:.2}, val r2: {epoch_val_r2:.2}'
                    io.cprint(outstr)

                    if epoch_val_r2 >= best_val_r2:
                        best_val_r2 = epoch_val_r2
                        best_val_rmse = epoch_val_rmse

                        if args.linear_classifier: 
                            torch.save(model_reg.state_dict(), f'checkpoints/{file_name}/models/model_reg_lc_limited_{args.limited_ratio}_{rep}.pth')
                        elif args.finetune: 
                            torch.save(model_reg.state_dict(), f'checkpoints/{file_name}/models/model_reg_ft_limited_{args.limited_ratio}_{rep}.pth')
                        else: 
                            torch.save(model_reg.state_dict(), f'checkpoints/{file_name}/models/model_reg_scratch_limited_{args.limited_ratio}_{rep}.pth')
                        
                        outstr = f'Best model RMSE: {best_val_rmse}, R2Score:{best_val_r2}'
                        io.cprint(outstr)


                        # Save predictions
                        gt = torch.cat(gt, dim=0).cpu().numpy()
                        predictions = torch.cat(predictions, dim=0).cpu().numpy()

                        validation_results = pd.DataFrame({
                            'Ground Truth': gt.flatten(),
                            'Predictions': predictions.flatten()
                        })

                        validation_results_file_name = f'validation_results_limited_ratio_{args.limited_ratio}_{rep}.csv'
                        validation_results.to_csv(os.path.join(args.output_path, validation_results_file_name), index=False)
                        print(f'Validation rsults saved to {validation_results_file_name}')

        val_r2_reps.append(best_val_r2)
        val_rmse_reps.append(best_val_rmse)
        io.cprint(f'Best validation RMSE: {best_val_rmse}, R2Score:{best_val_r2}')

        ### TESTING ###
        io.cprint(f'EVALUATION TEST SET:')

        # reset all metrics
        mae_val.reset()
        mse_val.reset()
        r2_val.reset()

        # load the best model (see validation)
        if args.finetune:
            model_reg.load_state_dict(torch.load(f'checkpoints/{file_name}/models/model_reg_ft_limited_{args.limited_ratio}_{rep}.pth'))
        elif args.linear_classifier:
            model_reg.load_state_dict(torch.load(f'checkpoints/{file_name}/models/model_reg_lc_limited_{args.limited_ratio}_{rep}.pth'))
        elif args.from_scratch:
            model_reg.load_state_dict(torch.load(f'checkpoints/{file_name}/models/model_reg_scratch_limited_{args.limited_ratio}_{rep}.pth'))
        model_reg.eval()    
        gt = []
        predictions = []
        with torch.no_grad():
            for batch, (data, labels) in enumerate(tqdm(test_loader)):
                data = data.to(device).float()
                data = data.permute(0, 2, 1)
                gt.append(labels.squeeze(-1).to(device))

                pred = model_reg(data)
                predictions.append(pred.cpu())

            # cat gt and predictions
            gt = torch.cat(gt, dim=0).to(device)
            predictions = torch.cat(predictions, dim=0).squeeze(-1).to(device)
            
            # Calculate metrics
            mae = mae_val(predictions, gt)
            mse = mse_val(predictions, gt)
            r2 = r2_val(predictions, gt)
            rmse = torch.sqrt(mse)
            mbe = torch.mean(predictions - gt)
            
            io.cprint(
                'TEST METRICS: \n'
                f'MAE: {mae:.2f}, \n'
                f'MSE: {mse:.2f}, \n'
                f'RMSE: {rmse:.2f}, \n'
                f'R2: {r2:.2f}, \n'
                f'MBE: {mbe:.2f}, \n'
            )

            # Save predictions
            gt = gt.cpu().numpy() 
            predictions = predictions.cpu().numpy()

            test_results = pd.DataFrame({
                'Ground Truth': gt.flatten(),
                'Predictions': predictions.flatten()
            })

            test_results_file_name = f'test_results_limited_ratio_{args.limited_ratio}_{rep}.csv'
            test_results.to_csv(os.path.join(args.output_path, test_results_file_name), index=False)
            print(f'Test results saved to {test_results_file_name}')   

            test_r2_reps.append(r2.cpu().numpy())
            test_rmse_reps.append(rmse.cpu().numpy())

            io.cprint(f'Best test RMSE: {rmse.cpu().numpy()}, R2Score:{r2.cpu().numpy()}')

    # Save all metrics for all repetitions  
    metrics_df = pd.DataFrame({
        'Repetition': np.arange(1, args.repetitions + 1),
        'Validation RMSE': val_rmse_reps,
        'Validation R2': val_r2_reps,
        'Test RMSE': test_rmse_reps,
        'Test R2': test_r2_reps
    })

    df_mean = pd.DataFrame(metrics_df.mean(axis=0).round(2))
    df_std = pd.DataFrame(metrics_df.std(axis=0).round(2))

    metrics_df = pd.concat([metrics_df, df_mean.T, df_std.T], axis=0).reset_index(drop=True)
    metrics_df.index = [f'rep_{i}' for i in range(1, args.repetitions + 1)] + ['mean', 'std']

    metrics_df.drop(columns=['Repetition'], inplace=True)


    metrics_df.to_csv(os.path.join(args.output_path, f'{args.limited_ratio}_allrepetitions.csv'), index=False)
    io.cprint(f'Experiment metrics saved to {args.output_path}{args.limited_ratio}_allrepetitions.csv')
    #io.cprint(f'Experiment metrics: \n{io.cprint(metrics_df)}')

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Completion Pre-training')
    parser.add_argument('--exp_name', type=str, default='exp_shapenet55_block', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--mask_ratio', type=float, default=0.7, help='masking ratio')
    parser.add_argument('--random', type=bool, default=False, metavar='N', help='random masking')
    parser.add_argument('--patch_size', type=int, default=64, help='patch size')

    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'ScanObjectNN_objectonly', 'ScanObjectNN_objectbg', 'ScanObjectNN_hardest', 'forage', 'forage_h5'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=251, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001, 0.1 if using sgd)')  # 0.00000001
    parser.add_argument('--momentum', type=float, default=0.7, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--limited_ratio', type=float, default=1.0,
                        help='dropout rate')

    parser.add_argument('--pretrained', type=bool, default=False, metavar='N',
                        help='Restore model from path')

    parser.add_argument('--finetune', type=bool, default=False, metavar='N',
                        help='Restore model from path')
    parser.add_argument('--linear_classifier', type=bool, default=False, metavar='N',
                        help='random mask')
    parser.add_argument('--from_scratch', type=bool, default=False, metavar='N')
    
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--repetitions', type=int, default=1, metavar='N',
                        help='number of repetitions for the experiment')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    file_name = 'mask_ratio_' + str(args.mask_ratio) + '/' + args.exp_name
    pretrain_model_path = './checkpoints/mask_ratio_' + str(args.mask_ratio) + '/' + args.exp_name + '/models/model_pretrain.pth'
    model_path = './checkpoints/mask_ratio_' + str(args.mask_ratio) + '/' + args.exp_name + '/models/model_reg_ft.pth'
    _init_()

    io = IOStream('checkpoints/' + file_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    # set seeds for torch cpu and numpy -> cuda already set above
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    train(args, io, file_name)