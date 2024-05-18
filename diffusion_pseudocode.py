import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Sinusoidal position embedding for time steps
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = np.concatenate((np.sin(emb), np.cos(emb)), axis=-1)
    return torch.tensor(emb, dtype=torch.float32)

# Define a simple 1D U-Net architecture (for illustration purposes)
class UNet1D(nn.Module):
    def __init__(self):
        super(UNet1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.time_embed = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Use Tanh to scale output between -1 and 1
        )
        
    def forward(self, x, t):
        x = self.encoder(x)
        # Add time-step embedding
        t_emb = self.time_embed(t).unsqueeze(-1)  # Expand dimensions to match
        x = x + t_emb
        x = self.decoder(x)
        return x

# Parameters
num_steps = 1000
embedding_dim = 32  # Dimension for time-step embedding
beta = np.linspace(0.0001, 0.02, num_steps)  # Noise schedule
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)  # Cumulative product of (1 - beta)

# Instantiate the model, loss function, and optimizer
model = UNet1D()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
def train_model(data, num_steps, alpha_bar, embedding_dim, epochs=100):
    for epoch in range(epochs):
        for _ in range(len(data)):  # Assuming data is a list of training examples
            x0 = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            t = np.random.randint(1, num_steps+1)
            t_emb = get_timestep_embedding(np.array([t]), embedding_dim)
            epsilon = torch.randn_like(x0)
            
            # Get alpha_bar values for step t
            sqrt_alpha_bar_t = np.sqrt(alpha_bar[t-1])
            sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar[t-1])
            
            # Create noisy data xt
            xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * epsilon
            
            # Predict the noise using the model with time-step embedding
            epsilon_theta = model(xt, t_emb)
            
            # Calculate the loss
            loss = criterion(epsilon_theta, epsilon)
            
            # Perform a gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example data
data = np.array([0.2, 0.5, 0.8])

# Training the model
train_model(data, num_steps, alpha_bar, embedding_dim, epochs=100)

# Forward process (for testing purpose)
def forward_process(data, num_steps, beta):
    noisy_data = [data]
    for t in range(num_steps):
        noise = np.random.normal(0, beta[t], data.shape)
        data = data + noise
        noisy_data.append(data)
    return noisy_data

# Reverse process (for testing purpose)
def reverse_process(noisy_data, num_steps, beta, embedding_dim):
    for t in reversed(range(num_steps)):
        t_emb = get_timestep_embedding(np.array([t+1]), embedding_dim)
        noise_pred = model(torch.tensor(noisy_data[t], dtype=torch.float32).unsqueeze(0).unsqueeze(0), t_emb).squeeze().detach().numpy()
        noisy_data[t-1] = noisy_data[t] - beta[t] * noise_pred
    return noisy_data[0]

# Forward process
noisy_data = forward_process(data, num_steps, beta)

# Reverse process (starting from final noisy data)
recovered_data = reverse_process(noisy_data, num_steps, beta, embedding_dim)

print("Original data:", data)
print("Recovered data:", recovered_data)
