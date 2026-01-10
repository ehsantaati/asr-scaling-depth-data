#!/usr/bin/env python
import argparse
import logging
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run training with multiple seeds")
    parser.add_argument("--config_path", type=str, required=True, help="Path to base config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,123,1234,12345,123456], help="List of seeds to run")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    
    args = parser.parse_args()
    
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # We need to extract the base run_name or exp_id from the config or filename
    # However, train.py handles run_name generation based on config.
    # We will override run_name via command line if possible, OR we supply arguments that modify it.
    # TrainConfig has `run_name`. simple_parsing allows overriding via --run_name.
    
    # We'll use subprocess to call train.py
    # Command: python train.py --config_path X --seed S --run_name NAME
    
    # Let's read the config first to see if there is an existing run name, but simple_parsing is easiest
    # to just rely on train.py's logic if we don't pass run_name?
    # No, we want distinct output folders. train.py uses run_name or output_dir/exp_id.
    # If we pass --run_name, train.py uses it.
    
    # Strategy:
    # Get the exp_id from the config filename (e.g. 001_s_1 from exps/001_s_1.yaml)
    exp_id = config_path.stem 
    
    for seed in args.seeds:
        logging.info(f"=== Running Experiment {exp_id} with Seed {seed} ===")
        
        # specific run name
        # We use just "seed_{seed}" so the output folder becomes outputs/.../exp_id/seed_{seed}_...
        # instead of repeating output/.../exp_id/exp_id_seed_{seed}_...
        run_name = f"seed_{seed}"
        
        cmd = [
            "poetry", "run", "python", "train.py",
            "--config_path", str(config_path),
            "--seed", str(seed),
            "--run_name", run_name
            # data_seed defaults to 42 in config, so it stays fixed unless we override it.
            # We explicitly WANT it fixed, so we don't pass it.
        ]
        
        logging.info(f"Executing: {' '.join(cmd)}")
        
        # Run
        # We set CUDA_VISIBLE_DEVICES via env
        env = dict(sys.modules['os'].environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
        
        try:
            subprocess.run(cmd, env=env, check=True)
            logging.info(f"=== Completed Seed {seed} ===\n")
        except subprocess.CalledProcessError as e:
            logging.error(f"!!! Failed run for seed {seed} !!!")
            logging.error(e)
            # Depending on requirement, we might want to stop or continue. 
            # Usually strict failure is better.
            sys.exit(1)

if __name__ == "__main__":
    main()
