import torch
from model import DiffusionModel
from torch.utils.data import DataLoader
import imageio
from maze3d.gen_maze_dataset_new import gen_dataset
from maze3d import maze
from maze3d import maze_env
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############ Utility Functions ############
def get_batch(color, pose, obs_size=12, batch_size=32, to_torch=True):
    img_obs = None
    pose_obs = None
    img_query = None
    pose_query = None
    for i in range(batch_size):
        batch_id = np.random.randint(0, color.shape[0])
        obs_id = np.random.randint(0, color.shape[1], size=obs_size)
        query_id = np.random.randint(0, color.shape[1])
        
        if img_obs is None:
            img_obs = color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])
            pose_obs = pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])
            img_query = color[batch_id:batch_id+1, query_id]
            pose_query = pose[batch_id:batch_id+1, query_id]
        else:
            img_obs = np.concatenate([img_obs, color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])], 0)
            pose_obs = np.concatenate([pose_obs, pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])], 0)
            img_query = np.concatenate([img_query, color[batch_id:batch_id+1, query_id]], 0)
            pose_query = np.concatenate([pose_query, pose[batch_id:batch_id+1, query_id]], 0)
    
    if to_torch:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_obs = torch.FloatTensor(img_obs).permute(0,3,1,2).to(device)
        pose_obs = torch.FloatTensor(pose_obs).to(device)
        img_query = torch.FloatTensor(img_query).permute(0,3,1,2).to(device)
        pose_query = torch.FloatTensor(pose_query).to(device)

        pose_obs = pose_obs.reshape(-1, obs_size, pose_obs.shape[1])
        pose_query = pose_query.reshape(-1, 1, pose_query.shape[1])

    return img_obs, pose_obs, img_query, pose_query

# Maze env
maze_obj = maze.MazeGridRandom2(obj_prob=0.3)
env = maze_env.MazeBaseEnv(maze_obj, render_res=(64,64), fov=80*np.pi/180)

# Training hyperparameters
diffusion_steps = 1000
max_epoch = 10
batch_size = 128
gen_dataset_size = 100
gen_dataset_step = 5000
eval_step = 1000
obs_size = 1

# Loading parameters
load_model = False
load_version_num = 1

# Create model and trainer
model = DiffusionModel(64*64, diffusion_steps, 3, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

if __name__ == "__main__":
    # Train model
    for steps in range(1000000):
        if steps % gen_dataset_step == 0:
            print("Generate Dataset ...")
            color_data, depth_data, pose_data = \
                gen_dataset(env, gen_dataset_size, samp_range=3.0, samp_size=100)
            color_data = (color_data.astype(float) / 255.)*2-1
            print("\nDone")
        
        x_obs, v_obs, x_q_gt, v_q = get_batch(color_data, pose_data, obs_size, batch_size)
        model.zero_grad()
        loss = model.get_loss(x_obs)
        loss.backward()
        optimizer.step()

        if steps % eval_step == 0:
            print("Step: " + str(steps).zfill(5) + " | loss: "+str(loss.detach().cpu().numpy()))
        
            gif_shape = [3, 3]
            sample_batch_size = gif_shape[0] * gif_shape[1]
            n_hold_final = 10

            # Generate samples from denoising process
            gen_samples = []
            x = torch.randn((sample_batch_size, 3, 64, 64)).to(device)
            sample_steps = torch.arange(model.t_range-1, 0, -1).to(device)
            for t in sample_steps:
                x = model.denoise_sample(x, t)
                if t % 50 == 0:
                    gen_samples.append(x[:,[2,1,0],:,:])
            for _ in range(n_hold_final):
                gen_samples.append(x[:,[2,1,0],:,:])
            gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
            gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

            # Process samples and save as gif
            gen_samples = (gen_samples * 255).type(torch.uint8)
            gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], 64, 64, 3)

            def stack_samples(gen_samples, stack_dim):
                gen_samples = list(torch.split(gen_samples, 1, dim=1))
                for i in range(len(gen_samples)):
                    gen_samples[i] = gen_samples[i].squeeze(1)
                return torch.cat(gen_samples, dim=stack_dim)

            gen_samples = stack_samples(gen_samples, 2)
            gen_samples = stack_samples(gen_samples, 2)

            imageio.mimsave(
                f"exp/pred_"+ str(steps).zfill(5) +".gif",
                list(gen_samples.cpu()),
                fps=5,
            )