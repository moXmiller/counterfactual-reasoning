print("Started")
import argparse
import torch
import numpy as np
import pandas as pd
import os
import csv
print("torch imported", flush = True)

from eval import get_model_from_run
from dataset import get_data_sampler
from tasks import mean_squared_error
from sde import get_sde_data_sampler
from tqdm import tqdm
import random
print("functions imported")

def retrieve_model(args, continuation = True):
    run_dir = "../models"
    if not os.path.exists(run_dir + "/" + "neurips"):
        os.makedirs(run_dir + "/" + "neurips")
    # df = read_run_dir(run_dir)
    if continuation: run_id, _ = retrieve_run_id(args)
    else: _, run_id = retrieve_run_id(args)
    print(f"run_id: {run_id}")

    # task_name = "counterfactual_regression"
    run_path = run_dir + "/" + "neurips" + "/" + run_id
    run_path = str(run_path)
    print(f"run_path: {run_path}")
    disentangled = bool(args.disentangled)
    weights = bool(args.weights)
    if disentangled or weights: raise NotImplementedError
    model, _ = get_model_from_run(run_path, disentangled=disentangled, weights=weights)
    
    print("model read in")
    return model


def model_to_device(model):
    if torch.cuda.is_available(): device = "cuda"
    else:
        device = "cpu"
    print(f"torch device: {device}")

    model.to(device)
    return model, device
    

def write_mse_sde(data_sampler, args):
    n_thetas = args.n_thetas
    o_vars = args.o_vars
    model_size = args.model_size
    ao = args.ao
    lamb, max_time = args.lamb, args.max_time
    n_points = args.n_points
    print("assigned arguments")

    model = retrieve_model(args, continuation = False)
    print("model retrieved")
    model, device = model_to_device(model)
    model.eval()
    print("model in eval mode")
    
    mean_row = {"stat": "mean", "model_size": model_size}
    opts_row = {"stat": "o_points", "model_size": model_size}

    eval_steps = 1000
    for itr in tqdm(range(eval_steps)):
        xs = data_sampler.complete_sde_dataset(n_thetas, o_vars, lamb, max_time, n_points, itr = itr, split = 'val')
        with torch.no_grad():
            pred, gt = model(xs.to(device), o_vars = o_vars)
        pred, gt = pred.unsqueeze(1), gt.unsqueeze(1)
        mse_thetas = torch.zeros(n_thetas)
        for theta_idx in range(n_thetas):
            mse_thetas[theta_idx] = mean_squared_error(pred[theta_idx,:,:].cpu().detach(), gt[theta_idx,:,:].cpu().detach()) # theta_hat^*

        mean_row[str(itr)] = torch.mean(mse_thetas)
        opts_row[str(itr)] = data_sampler.o_points

    mse_rows = [mean_row, opts_row]
    csv_field_names = [str(e) for e in range(eval_steps)]
    csv_field_names = ["stat", "model_size"] + csv_field_names
    filename = f"eval/mse/context_length_sde_{model_size}{'_ao' if ao else ''}.csv"

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(mse_rows)
    print(f"Successfully written .csv file for {model_size}!")


def write_mse(data_sampler, args):
    n_thetas = args.n_thetas
    o_points = args.o_points
    o_vars = args.o_vars
    model_size = args.model_size
    ao = args.ao
    theta_dist = args.theta_dist
    transformation = args.transformation
    train_steps = args.train_steps

    model = retrieve_model(args, continuation = False)
    
    model, device = model_to_device(model)
    model.eval()
    
    n_points = o_points
    
    mean_row = {"stat": "mean", "model_size": model_size, "attention_only": ao}
    q025_row = {"stat": "q025", "model_size": model_size, "attention_only": ao}
    q975_row = {"stat": "q975", "model_size": model_size, "attention_only": ao}

    eval_steps = 100
    for p in tqdm(range(1, n_points + 1)):
        mse_itr = torch.zeros(eval_steps)
        for itr in range(eval_steps):
            xs = data_sampler.complete_dataset(n_thetas, p, o_vars, itr = itr, split = 'val', continuation=False, transformation = args.transformation)
            with torch.no_grad():
                pred, gt = model(xs.to(device), o_vars = o_vars)
            pred, gt = pred.unsqueeze(1), gt.unsqueeze(1)
            mse_thetas = torch.zeros(n_thetas)
            for theta_idx in range(n_thetas):
                mse_thetas[theta_idx] = mean_squared_error(pred[theta_idx,:,:].cpu().detach(), gt[theta_idx,:,:].cpu().detach()) # theta_hat^*
            mse_itr[itr] = torch.mean(mse_thetas)
        
        quantile_indices = random.choices(range(eval_steps), k = eval_steps)
        quantiles = torch.tensor(np.quantile(mse_itr[quantile_indices].numpy(), q=[0.025, 0.975]))

        mean_mse = torch.mean(mse_itr).item()
        lower_q = max(0, 2 * mean_mse - torch.mean(quantiles[1]).item())
        upper_q = max(0, 2 * mean_mse - torch.mean(quantiles[0]).item())

        mean_row[str(p)] = mean_mse
        q025_row[str(p)] = lower_q
        q975_row[str(p)] = upper_q

    mse_rows = [mean_row, q025_row, q975_row]
    csv_field_names = [str(p) for p in range(1, n_points + 1)]
    csv_field_names = ["stat", "model_size", "attention_only"] + csv_field_names
    filename = f"eval/mse/context_length_{model_size}{'_ao' if ao else ''}{'_eval_on_n' if (theta_dist == 'norm') else ''}.csv"
    if model_size == "eightlayer" and transformation != "addlin": filename = f"eval/mse/context_length_{model_size}{'_ao' if ao else ''}_{transformation}{'_eval_on_n' if (theta_dist == 'norm') else ''}.csv"
    if model_size == "standard" and train_steps != 200000: filename = f"eval/mse/context_length_{model_size}{'_ao' if ao else ''}_{train_steps}{'_eval_on_n' if (theta_dist == 'norm') else ''}.csv"

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(mse_rows)
    print(f"Successfully written .csv file for {model_size}!")


def variance_extensions(data_sampler, args):
    eval_steps = 100
    n_thetas = args.n_thetas
    o_vars = args.o_vars
    o_points = 1
    o_dims = args.o_dims
    transformation = args.transformation

    finals = []
    for step in tqdm(range(eval_steps)):
        xs = data_sampler.complete_dataset(n_thetas, o_points, o_vars, transformation = transformation, itr = step, split = 'val', continuation=False)
        final = xs[:,-1,:].tolist()[0]
        # if step == 0: print(final)
        finals.append(final)
    
    return torch.var(torch.tensor(finals))


def write_attentions(data_sampler, args):
    model_size = args.model_size
    if model_size == "tiny": n_layer, n_head = 3, 2
    elif model_size == "small": n_layer, n_head = 6, 4
    elif model_size == "standard": n_layer, n_head = 12, 8
    elif "twolayer" in model_size: n_layer, n_head = 2, 1
    elif "threelayer" in model_size: n_layer, n_head = 3, 1
    elif "fourlayer" in model_size: n_layer, n_head = 4, 1
    elif "eightlayer" in model_size: n_layer, n_head = 8, 1
    elif "twelvelayer" in model_size: n_layer, n_head = 12, 1
    elif "sixteenlayer" in model_size: n_layer, n_head = 16, 1
    else: NotImplementedError
    
    n_thetas = args.n_thetas
    o_points = args.o_points
    o_vars = args.o_vars

    if continuation: raise NotImplementedError

    ao = args.ao
    position = args.position
    itr = args.itr
    continuation = False

    model = retrieve_model(args, continuation = continuation)
    
    model, device = model_to_device(model)

    if args.data == "gaussian":
        xs = data_sampler.complete_dataset(n_thetas, o_points, o_vars, itr = itr, split = 'val', continuation=continuation)
    elif args.data == "sde": 
        lamb, max_time, n_points = args.lamb, args.max_time, args.n_points
        xs = data_sampler.complete_sde_dataset(n_thetas, o_vars, lamb, max_time, n_points, itr = itr, split = 'val')
    with torch.no_grad():
        _, _, attentions = model(xs.to(device), o_vars = o_vars, output_attentions = True)
    
    for l in range(0, n_layer):
        print("layer", l)
        for h in range(0, n_head):
            if position == -1: 
                att = torch.mean(attentions[l][:,h,:,:], dim = 0)
                att = pd.DataFrame(att.cpu().detach().numpy())
            else: att = pd.DataFrame(attentions[l][position,h,:,:].cpu().detach().numpy())
            if position == -1:
                attention_path = f"eval/attentions/attentions_{model_size}_{l}_layer_{h}_head{'_ao' if ao else ''}_{itr}_m{'_sde' if args.data == 'sde' else ''}.csv"
            else:
                attention_path = f"eval/attentions/attentions_{model_size}_{l}_layer_{h}_head{'_ao' if ao else ''}_itr{itr}_pos{position}{'_sde' if args.data == 'sde' else ''}.csv"            
            att.to_csv(attention_path, index = False)
    print(f"attentions successfully written to .csv for {model_size}")


def sde_data_for_plot(data_sampler, args):
    n_thetas = args.n_thetas
    o_vars = args.o_vars
    model_size = args.model_size
    ao = args.ao
    itr = args.itr
    dimension_idx = args.dim_index
    lamb, max_time = args.lamb, args.max_time
    position = args.position
    if args.position == -1: 
        position = n_thetas - 1
        print("Last element selected")
    n_points = args.n_points
    print("assigned arguments")

    model = retrieve_model(args, continuation = False)
    print("model retrieved")
    model, device = model_to_device(model)
    model.eval()
    print("model in eval mode")

    obs_x_row = {"quantity": "obs_x"}
    obs_y_row = {"quantity": "obs_y"}
    pred_x_row = {"quantity": "pred_x"}
    pred_y_row = {"quantity": "pred_y"}
    gt_x_row = {"quantity": "gt_x"}
    gt_y_row = {"quantity": "gt_y"}
    evnt_t_row = {"quantity": "event_times"}

    xs = data_sampler.complete_sde_dataset(n_thetas, o_vars, lamb, max_time, n_points, itr = itr, split = 'val')
    with torch.no_grad():
        pred, gt = model(xs.to(device), o_vars = o_vars)
    # include x_init_cf, y_init_cf

    event_times = data_sampler.event_times
    o_points = data_sampler.o_points

    xs_obs = xs[:, :(o_points * o_vars), :].to(device)
    obs_x = xs_obs[position, ::2, dimension_idx]
    obs_y = xs_obs[position, 1::2, dimension_idx]

    cf_init = xs[:, (o_points * o_vars + 1):(o_points * o_vars + o_vars + 1), :].to(device)
    pred = torch.cat([cf_init, pred], dim=1)
    pred_x = pred[position, ::2, dimension_idx]
    pred_y = pred[position, 1::2, dimension_idx]
    gt = torch.cat([cf_init, gt], dim = 1)
    gt_x = gt[position, ::2, dimension_idx]
    gt_y = gt[position, 1::2, dimension_idx]

    for p in range(o_points):
        obs_x_row[str(p)] = obs_x[p].cpu().detach().numpy()
        obs_y_row[str(p)] = obs_y[p].cpu().detach().numpy()
        pred_x_row[str(p)] = pred_x[p].cpu().detach().numpy()
        pred_y_row[str(p)] = pred_y[p].cpu().detach().numpy()
        gt_x_row[str(p)] = gt_x[p].cpu().detach().numpy()
        gt_y_row[str(p)] = gt_y[p].cpu().detach().numpy()
        evnt_t_row[str(p)] = event_times[p].cpu().detach().numpy()


    csv_field_names = [str(p) for p in range(o_points)]
    csv_field_names = ["quantity"] + csv_field_names
    filename = f"eval/mse/prediction_chart_sde_{model_size}{'_ao' if ao else ''}.csv"

    prediction_rows = [obs_x_row, obs_y_row, pred_x_row, pred_y_row, gt_x_row, gt_y_row, evnt_t_row]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_field_names) 
        writer.writeheader() 
        writer.writerows(prediction_rows)
    print(f"Successfully written .csv file for {model_size}!")


# the models are not provided in the supplementary material
# provided code suffices to train models from scratch
# training on one NVIDIA GeForce RTX 3090 GPU takes between 10 minutes and 3 hours
# most models can be trained in 30 to 60 minutes
def retrieve_run_id(args, lin_reg = False):
    if lin_reg: raise NotImplementedError
    model_size = args.model_size
    data_arg = args.data
    ao = args.ao
    transformation = args.transformation
    if data_arg == "gaussian":
        if not ao:
            if model_size == "standard":
                cont_run_id = None
                cf_run_id = "put-in-run-id"
                # run id has the form aB3dE6gH-9jK2-mN5p-Q8sT-1vW4yZ7bC0eF
            else: raise NotImplementedError
        else:
            pass
    elif data_arg == "sde":
        if not ao:
            if model_size == "standard":
                cont_run_id = None
                cf_run_id = "put-in-run-id"
            else: pass
        else:
            pass
    return cont_run_id, cf_run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser for dataset arguments')

    parser.add_argument('--o_dims',         type=int, default=5, help="o_dims for dataset setup")
    parser.add_argument('--o_vars',         type=int, default=2, help="o_vars for dataset setup")
    parser.add_argument('--o_points',       type=int, default=50, help="o_points for dataset setup")
    parser.add_argument('--n_points',       type=int, default=60, help="n_points for sde dataset setup")
    parser.add_argument('--n_thetas',       type=int, default=64, help="n_thetas for dataset setup")
    parser.add_argument('--lamb',           type=int, default=200, help="lambda parameter for poisson process")
    parser.add_argument('--max_time',       type=int, default=.1, help="maximum time for brownian motion: in expectation, lamb * max_time = o_points")
    parser.add_argument('--dag_type',       type=str, default="only_parent", help="dag_type for dataset setup")

    parser.add_argument('--data',           type=str, default="gaussian", help="data type for get_data_sampler")
    parser.add_argument('--train_steps',    type=int, default=200000, help="Number of steps the model is trained on:    required for eval_seeds_dict")
    parser.add_argument('--eval_steps',     type=int, default=1000, help="Number of evaluation steps:                   required for eval_seeds_dict")
    parser.add_argument('--diversity',      type=int, default=128000000, help="Diversity")
    parser.add_argument('--theta_dist',     type=str, default="uniform", help="Distribution of theta")
    parser.add_argument('--transformation', type=str, default="addlin", help="Transformation of complete dataset")
    parser.add_argument('--batch_size',     type=int, default=64, help="similar to n_thetas")
    parser.add_argument('--model_size',     type=str, default="standard", help="model_size to evaluate, one of [tiny, small, standard, fourlayer, eightlayer]")
    parser.add_argument('--ao',             type=int, default=0, help="attention_only argument for model extraction")
    parser.add_argument('--disentangled',   type=int, default=0, help="work on the disentangled transformer")
    parser.add_argument('--weights',        type=int, default=0, help="retrieve weight matrices")
    parser.add_argument('--position',       type=int, default=-1, help="position to compute attentions for, default: -1, mean")
    parser.add_argument('--itr',            type=int, default=0, help="iteration to compute attentions for, default: 0, first seed sampling iteration")
    parser.add_argument('--dim_index',      type=int, default=0, help="dimension to compute SDE predictions for, default: 0, first dimension")
    args = parser.parse_args()

    o_dims = args.o_dims
    kwargs = {"dag_type": args.dag_type}

    if args.data == "gaussian":
        data_sampler = get_data_sampler(args, o_dims, **kwargs)
        # write_attentions(data_sampler, args)
        # write_mse(data_sampler, args)
        var = variance_extensions(data_sampler, args)
        print(args.model_size, var)

    if args.data == "sde":
        print("starting data sampler")
        data_sampler = get_sde_data_sampler(args, o_dims, **kwargs)
        print("finalizing data sampler")
        # write_mse_sde(data_sampler, args)
        # write_attentions(data_sampler, args)
        sde_data_for_plot(data_sampler, args)