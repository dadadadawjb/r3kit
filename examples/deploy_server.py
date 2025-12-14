import os
import json
import time
from rich import print
from tap import Tap
import torch

from r3kit.utils.buffer import ObsBuffer, ActBuffer

from policy import Policy, preprocess_obs, postprocess_act

class ArgumentParser(Tap):
    cfg_path: str = './ckpt/config.yaml'
    ckpt_path: str = './ckpt/best_model.pth'

    num_obs: int = 1
    num_actions: int = 20
    sleep_time: float = 0.01

    meta_path: str = './.meta/'


def main(args:ArgumentParser):
    # initialize policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = Policy(args.cfg_path).to(device)
    policy.load_state_dict(torch.load(args.ckpt_path, map_location=device), strict=False)
    
    # initialize buffers
    with open(os.path.join(args.meta_path, 'obs_dict.json'), 'r') as f:
        obs_dict = json.load(f)
    with open(os.path.join(args.meta_path, 'act_dict.json'), 'r') as f:
        act_dict = json.load(f)
    obs_buffer = ObsBuffer(num_obs=args.num_obs, obs_dict=obs_dict, create=False)
    act_buffer = ActBuffer(num_act=args.num_actions, act_dict=act_dict, create=False)
    print("=========> Initialized")

    # rollout
    idx = 0
    policy.eval()
    with torch.inference_mode():
        while True:
            # get obs
            while not obs_buffer.getf():
                time.sleep(args.sleep_time)
            o = obs_buffer.getn()
            if len(o) < args.num_obs:
                o = [o[0]] * (args.num_obs - len(o)) + o
            obs_buffer.setf(False)
            print(f"=========> Get obs {idx}")

            # preprocess obs
            o = preprocess_obs(o)

            # get act
            a = policy(o)

            # postprocess act
            a = postprocess_act(a)

            # set act
            act_buffer.addn(a)
            act_buffer.setf(True)
            print(f"=========> Add act {idx}")
            idx += 1


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
