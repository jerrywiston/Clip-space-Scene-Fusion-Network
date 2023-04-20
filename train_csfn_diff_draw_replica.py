import numpy as np
import cv2
import os
import json
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import configparser
import config_handle
import utils
from core_csfn.csfn_diff_draw import CSFN
from dataset_replica import load_replica

############ Utility Functions ############
def get_batch(color, pose, obs_size=12, batch_size=32, device="cuda"):
    img_obs = None
    pose_obs = None
    img_query = None
    pose_query = None
    for i in range(batch_size):
        batch_id = np.random.randint(0, color.shape[0])
        obs_id = np.random.randint(0, color.shape[1], size=obs_size)
        query_id = np.random.randint(0, color.shape[1])
        
        if img_obs is None:
            img_obs = color[batch_id:batch_id+1, obs_id]
            pose_obs = pose[batch_id:batch_id+1, obs_id]
            img_query = color[batch_id:batch_id+1, query_id:query_id+1]
            pose_query = pose[batch_id:batch_id+1, query_id:query_id+1]
        else:
            img_obs = torch.cat([img_obs, color[batch_id:batch_id+1, obs_id]], 0)
            pose_obs = torch.cat([pose_obs, pose[batch_id:batch_id+1, obs_id]], 0)
            img_query = torch.cat([img_query, color[batch_id:batch_id+1, query_id:query_id+1]], 0)
            pose_query = torch.cat([pose_query, pose[batch_id:batch_id+1, query_id:query_id+1]], 0)
    
    img_obs = img_obs.reshape([-1, img_obs.shape[-3], img_obs.shape[-2], img_obs.shape[-1]]).to(device) 
    pose_obs = pose_obs.to(device)  
    img_query = img_query.reshape([-1, img_query.shape[-3], img_query.shape[-2], img_query.shape[-1]]).to(device) 
    pose_query = pose_query.to(device) 

    return img_obs, pose_obs, img_query, pose_query

########

def draw_query_maze(net, color, pose, obs_size=3, row_size=32, gen_size=10, shuffle=False, border=[1,4,3]):
    img_list = []
    for it in range(gen_size):
        img_row = []
        x_obs, v_obs, x_query_gt, v_query = get_batch(color, pose, obs_size, row_size, device)
        img_size = (x_obs.shape[-2], x_obs.shape[-1])
        vsize = v_obs.shape[-1]
        with torch.no_grad():
            #print(x_obs.shape, v_obs.shape, v_query.shape)
            x_samp_draw, x_samp_diff = net.sample(x_obs, v_obs, v_query)
            x_samp_draw = x_samp_draw.detach().permute(0,2,3,1).cpu().numpy()
            x_samp_diff = x_samp_diff.detach().permute(0,2,3,1).cpu().numpy()
        
        for j in range(x_samp_draw.shape[0]):
            x_np = []
            bscale = int(img_size[1]/64)
            for i in range(obs_size):
                x_np.append(x_obs[obs_size*j+i].detach().permute(1,2,0).cpu().numpy())
                if i < obs_size-1:
                    x_np.append(np.ones([img_size[0],border[0]*bscale,3]))
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_query_gt.detach().permute(0,2,3,1).cpu().numpy()[j])
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_samp_draw[j])
            x_np.append(x_samp_diff[j])
            x_np = np.concatenate(x_np, 1)
            img_row.append(x_np)

            if j < row_size:
                img_row.append(np.ones([border[2]*bscale,x_np.shape[1],3]))
        
        img_row = np.concatenate(img_row, 0) * 255
        img_row = cv2.cvtColor(img_row.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_list.append(img_row.astype(np.uint8))
        fill_size = len(str(gen_size))
        print("\rProgress: "+str(it+1).zfill(fill_size)+"/"+str(gen_size), end="")
    print()
    return img_list

def draw_query_maze_layer(net, color, pose, obs_size=3, row_size=32, gen_size=10, shuffle=False, border=[1,4,3]):
    img_list = []
    for it in range(gen_size):
        img_row = []
        x_obs, v_obs, x_query_gt, v_query = get_batch(color, pose, obs_size, row_size, device)
        img_size = (x_obs.shape[-2], x_obs.shape[-1])
        vsize = v_obs.shape[-1]
        with torch.no_grad():
            for i in range(6):
                x_samp_draw_, x_samp_diff_ = net.sample(x_obs, v_obs, v_query, i)
                if i == 0:
                    x_samp_draw = x_samp_draw_.detach().permute(0,2,3,1).cpu().numpy()
                    x_samp_diff = x_samp_diff_.detach().permute(0,2,3,1).cpu().numpy()
                else:
                    x_samp_draw = np.concatenate((x_samp_draw, x_samp_draw_.detach().permute(0,2,3,1).cpu().numpy()),2)
                    x_samp_diff = np.concatenate((x_samp_diff, x_samp_diff_.detach().permute(0,2,3,1).cpu().numpy()),2)
        
        for j in range(x_samp_draw.shape[0]):
            x_np = []
            bscale = int(img_size[1]/64)
            for i in range(obs_size):
                x_np.append(x_obs[obs_size*j+i].detach().permute(1,2,0).cpu().numpy())
                if i < obs_size-1:
                    x_np.append(np.ones([img_size[0],border[0]*bscale,3]))
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_query_gt.detach().permute(0,2,3,1).cpu().numpy()[j])
            x_np.append(np.ones([img_size[0],border[1]*bscale,3]))
            x_np.append(x_samp_draw[j])
            x_np.append(x_samp_diff[j])
            x_np = np.concatenate(x_np, 1)
            img_row.append(x_np)

            if j < row_size:
                img_row.append(np.ones([border[2]*bscale,x_np.shape[1],3]))
        
        img_row = np.concatenate(img_row, 0) * 255
        img_row = cv2.cvtColor(img_row.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_list.append(img_row.astype(np.uint8))
        fill_size = len(str(gen_size))
        print("\rProgress: "+str(it+1).zfill(fill_size)+"/"+str(gen_size), end="")
    print()
    return img_list

########

def eval_maze(net, color, pose, obs_size=3, max_batch=1000, render_layers=-1, shuffle=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rmse_record = []
    mae_record = []
    ce_record = []
    for it in range(max_batch):
        img_row = []
        x_obs, v_obs, x_query_gt, v_query = get_batch(color, pose, obs_size, 32, device)
        img_size = (x_obs.shape[-2], x_obs.shape[-1])
        vsize = v_obs.shape[-1]
        with torch.no_grad():
            x_samp_draw, x_samp_diff = net.sample(x_obs, v_obs, v_query, render_layers)
            # rmse
            mse_batch = (x_samp_diff*255 - x_query_gt*255)**2
            rmse_batch = torch.sqrt(mse_batch.mean([1,2,3])).cpu().numpy()
            rmse_record.append(rmse_batch)
            # mae
            mae_batch = torch.abs(x_samp_diff*255 - x_query_gt*255)
            mae_batch = mae_batch.mean([1,2,3]).cpu().numpy()
            mae_record.append(mae_batch)
        fill_size = len(str(max_batch))
        print("\rProgress: "+str(it+1).zfill(fill_size)+"/"+str(max_batch), end="")
    
    print("\nDone~~")
    rmse_record = np.concatenate(rmse_record, 0)
    rmse_mean = rmse_record.mean()
    rmse_std = rmse_record.std()
    mae_record = np.concatenate(mae_record, 0)
    mae_mean = mae_record.mean()
    mae_std = mae_record.std()
    return {"rmse":[float(rmse_mean), float(rmse_std)],
            "mae" :[float(mae_mean), float(mae_std)],}

def eval(color_data_test, pose_data_test, net, img_path="experiments/eval/", row_size=10):
    ##
    max_obs_size = 16
    obs_size = 10
    gen_size = 10
    ##
    print("------------------------------")


    # Test
    print("Generate testing image ...")
    fname = img_path+"test.png"
    canvas = draw_query_maze_layer(net, color_data_test, pose_data_test, obs_size=obs_size, row_size=row_size, gen_size=1, shuffle=True)[0]
    cv2.imwrite(fname, canvas)

############ Parameter Parsing ############
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', nargs='?', type=str, default="maze" ,help='Experiment name.')

args = lambda: None
args.exp_name = parser.parse_args().exp_name
args.img_size = (64, 64)#(128, 128)
args.view_size = (16, 16)#(32, 32)
args.depth_size = 6
args.pose_size = 12
args.emb_size = 16
args.cell_size = 128
args.fusion_type = "ocm"
args.loss_type = "MSE"
args.total_steps = 1600000
args.min_obs_size = 16
args.max_obs_size = 24

# Print Training Information
print("Experiment Name: %s"%(args.exp_name))
print("View Cell Size: (%d, %d)"%(args.view_size[0], args.view_size[1]))
print("Depth Size: %d"%(args.depth_size))
print("Pose Size: %d"%(args.pose_size))
print("Embedding Size: %d"%(args.emb_size))
print("Celle Size: %d"%(args.cell_size))
print("Fusion Type: %s"%(args.fusion_type))

print("Load ReplicA Dataset ...")
color_data_train, pose_data_train = load_replica("E:/ml-gsn/data/replica_all/train", args.img_size)

print("\nDone")

############ Create Folder ############
now = datetime.datetime.now()
tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
exp_path = "experiments/"
model_name = args.exp_name + "_%d_%d"%(args.depth_size, args.cell_size)
model_path = exp_path + tinfo + "_" + model_name + "/"

img_path = model_path + "img/"
save_path = model_path + "save/"
if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

############ Networks ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CSFN(args.view_size, args.depth_size, args.pose_size, args.emb_size, args.cell_size, args.fusion_type).to(device)
params = list(net.parameters())
opt = optim.Adam(params, lr=5e-5, betas=(0.5, 0.999))

# ------------ Loss Function ------------
if args.loss_type == "MSE":
    criterion = nn.MSELoss()
elif args.loss_type == "MAE":
    criterion = nn.L1Loss()
elif args.loss_type == "CE":
    creterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()
    
# ------------ Prepare Variable ------------
img_path = model_path + "img/"
save_path = model_path + "save/"
train_record = {"loss_query":[]}
eval_record = []
best_eval = 999999
steps = 0
epochs = 0
eval_step = 1000
zfill_size = len(str(args.total_steps))
batch_size = 16
gen_data_size = 100
gen_dataset_iter = 1000
samp_field = 3.0
train_diff_iter = 0#10000
test = False

if not test:
    while(True):   
        # ------------ Get data (Random Observation) ------------
        obs_size = np.random.randint(args.min_obs_size,args.max_obs_size)
        x_obs, v_obs, x_query_gt, v_query = get_batch(color_data_train, pose_data_train, obs_size, batch_size, device)
        
        # ------------ Forward ------------
        net.zero_grad()
        kl_loss, rec_loss, draw_loss, diff_loss, x_rec = net(x_obs, x_query_gt, v_obs, v_query)
        rec = [float(kl_loss.detach().cpu().numpy()),float(rec_loss.detach().cpu().numpy()),
            float(draw_loss.detach().cpu().numpy()),float(diff_loss.detach().cpu().numpy())]
                
        # ------------ Train ------------
        if steps > train_diff_iter:
            loss = draw_loss + diff_loss
        else:
            loss = draw_loss
        loss.backward()
        opt.step()
        steps += 1

        # ------------ Print Result ------------
        if steps % 100 == 0:
            print("[Ep %s/%s] kl_loss: %f, draw_loss: %f, diff_loss: %f"%(str(steps), str(args.total_steps), rec[0], rec[2], rec[3]))

        # ------------ Output Image ------------
        if steps % eval_step == 0:
            ##
            obs_size = 10
            gen_size = 5
            ##
            print("------------------------------")
            print("Generate Test Data ...")
            color_data_test, pose_data_test = load_replica("E:/ml-gsn/data/replica_all/test", args.img_size)
            print("Done!!")
            # Train
            print("Generate image ...")
            fname = img_path+str(int(steps/eval_step)).zfill(4)+"_train.png"
            canvas = draw_query_maze(net, color_data_train, pose_data_train, obs_size=obs_size, row_size=5, gen_size=1, shuffle=True)[0]
            cv2.imwrite(fname, canvas)
            # Test
            print("Generate testing image ...")
            fname = img_path+str(int(steps/eval_step)).zfill(4)+"_test.png"
            canvas = draw_query_maze(net, color_data_test, pose_data_test, obs_size=obs_size, row_size=5, gen_size=1, shuffle=True)[0]
            cv2.imwrite(fname, canvas)

            # ------------ Training Record ------------
            train_record["loss_query"].append(rec[0])
            print("Dump training record ...")
            with open(model_path+'train_record.json', 'w') as file:
                json.dump(train_record, file)

            # ------------ Evaluation Record ------------
            #print("Evaluate Training Data ...")
            #eval_results_train = eval_maze(net, color_data, pose_data, obs_size=6, max_batch=400, shuffle=False)
            print("Evaluate Testing Data ...")
            eval_results_test = eval_maze(net, color_data_test, pose_data_test, obs_size=6, max_batch=1, shuffle=False)
            #eval_record.append({"steps":steps, "train":eval_results_train, "test":eval_results_test})
            eval_record.append({"steps":steps, "test":eval_results_test})
            print("Dump evaluation record ...")
            with open(model_path+'eval_record.json', 'w') as file:
                json.dump(eval_record, file)

            # ------------ Save Model ------------
            if steps%100000 == 0:
                print("Save model ...")
                torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(zfill_size) + ".pth")
                    
            # Apply RMSE as the metric for model selection.
            if eval_results_test["rmse"][0] < best_eval:
                best_eval = eval_results_test["rmse"][0]
                print("Save best model ...")
                torch.save(net.state_dict(), save_path + "model.pth")
            print("Best Test RMSE:", best_eval)
            print("------------------------------")

        if steps >= args.total_steps:
            print("Save final model ...")
            torch.save(net.state_dict(), save_path + "model_" + str(steps).zfill(zfill_size) + ".pth")
            break
else:
    net.load_state_dict(torch.load(args.exp_name))
    #eval(color_data_test, pose_data_test, net)