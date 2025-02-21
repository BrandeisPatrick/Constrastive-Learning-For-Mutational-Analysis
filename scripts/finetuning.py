import sys
import os
from functools import partial
import argparse
import torch
import logging
import wandb

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from my_project import (
    get_config,
    Model_STT4,
    Trainer,
    Trainer_stt4_enc, 
    Model_STT4_encoder, 
    create_dataloaders, 
    print_parameters, 
    print_model_architecture 
)

from my_project.utils.load_model import load_model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Model_STT4")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config YAML file (default: config.yaml)'
    )
    args = parser.parse_args()

    # 1) Load YAML into our Config dataclass
    config = get_config(args.config)

    # 1.1) Ensure the save_dir exists
    try:
        os.makedirs(config.save_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating save directory '{config.save_dir}': {e}")
        sys.exit(1)

    # 3) Initialize logging
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(config.save_dir, 'finetune.log'))
            ]
        )
    except Exception as e:
        print(f"Error initializing logging: {e}")
        sys.exit(1)

    # 4) Initialize wandb with selective config logging
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        # Only log essential config parameters
        config={
            "learning_rate": config.lr,
            "optimizer_type": config.optimizer_type,
            "scheduler_type": config.scheduler_type,
            "max_epochs": config.max_epochs,
            "seed": config.seed,
            # Add other essential parameters as needed
        },
        reinit=True
    )

    # Optionally, log the config file path for reference
    wandb.config.update({"config_file": args.config}, allow_val_change=True)

    # Set random seed
    torch.manual_seed(config.seed)
    if config.default == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    # Print configuration parameters
    print_parameters(config)

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Initialize Model and move it to the correct device first
    model = Model_STT4_encoder(args=config).to(config.default)
    
    # Optionally resume from checkpoint BEFORE creating the optimizer
    if config.resume_checkpoint is not None:
        if not os.path.isfile(config.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {config.resume_checkpoint}")
        logging.info(f"Resuming training from checkpoint: {config.resume_checkpoint}")
        # Load the model from the checkpoint and move it to the device
        model = load_model(config, checkpoint_path=config.resume_checkpoint, load_encoder_only=False, model=model, strict_loading=False)
        model = model.to(config.default)

    # Print model architecture
    print_model_architecture(model)

    # Limit wandb.watch to log only gradients and increase log frequency
    wandb.watch(model, log="gradients", log_freq=10)  # Changed from "all" and log_freq=123

    # Initialize Optimizer using the (possibly resumed) model parameters
    if config.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    # Initialize Scheduler
    if config.scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif config.scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif config.scheduler_type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif config.scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    elif config.scheduler_type == "None":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")

    # (Optional) At this point you might also want to log the optimizer's parameter devices to confirm:
    for i, group in enumerate(optimizer.param_groups):
        for param in group['params']:
            logging.debug(f"Optimizer group {i} parameter device: {param.device}")

    # Initialize Trainer (pass the wandb run for logging)
    trainer = Trainer_stt4_enc(
        model=model,
        args=config,
        optimizer=optimizer,
        scheduler=scheduler,
        wandb_run=wandb.run
    )

    # Train
    trainer.train(train_loader, val_loader)

    # Evaluate on Test
    best_checkpoint = os.path.join(config.save_dir, f"model_epoch_{config.max_epochs}.pt")
    trainer.test(test_loader, best_checkpoint)

    # Optionally, save the best model locally without logging to W&B
    # Uncomment the following lines if you still want to save the model locally
    # torch.save(model.state_dict(), best_checkpoint)
    # logging.info(f"Best model saved locally at {best_checkpoint}")

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()