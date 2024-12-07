import torch
from simple_diffusion import SimpleDiffusion

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: torch.nn.Module,
               device: torch.device,
               sd: SimpleDiffusion):

    running_loss = 0.
    last_loss = 0.

    model.train()

    for i, (images, _) in enumerate(train_dataloader):
        timestep_input = torch.randint(1, sd.get_timesteps(), (images.shape[0],), dtype=torch.long)
        timestep_input_view = timestep_input.view(images.shape[0], 1, 1, 1)

        images = images.to(device).float()

        samples, noises = sd.forward_diffusion(images, timestep_input_view)
        
        optimizer.zero_grad()
        outputs = model(samples, timestep_input)
        loss = criterion(outputs, noises)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss /10
            print(f'batch {i+1} loss: {last_loss}')
            running_loss = 0.

    return last_loss

def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: torch.device,
              sd: SimpleDiffusion):

    running_vloss = 0.
    model.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(test_dataloader):
            timestep_input = torch.randint(1, sd.get_timesteps(), (images.shape[0],), dtype=torch.long)
            timestep_input_view = timestep_input.view(images.shape[0], 1, 1, 1)
    
            images = images.to(device).float()
    
            samples, noises = sd.forward_diffusion(images, timestep_input_view)
    
            outputs = model(samples, timestep_input)
    
            vloss = criterion(outputs, noises)
            running_vloss += vloss.item()
                
    avg_loss = running_vloss / (i + 1)
    return avg_loss

def train(epochs: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          model_path: str,
          device: torch.device,
          sd: SimpleDiffusion):

    best_vloss = 1000000

    for epoch in range(1,epochs+1):
        print(f'EPOCH {epoch}')
        avg_loss = train_step(model, train_dataloader, optimizer, criterion, device, sd)
        avg_vloss = test_step(model, test_dataloader, criterion, device, sd)
        print(f'LOSS train {avg_loss} test {avg_vloss}')

        if avg_vloss<best_vloss:
            best_vloss = avg_vloss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_vloss
                }, f'{model_path}_{epoch}.pth')
