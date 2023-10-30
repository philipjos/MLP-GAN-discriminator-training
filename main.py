import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchaudio
import torchaudio.transforms as transforms
import os
import sys

# Defining the properties of the elements in the target space.
target_sample_rate = 44100
target_length_seconds = 1
target_length_samples = target_sample_rate * target_length_seconds
# ------------------------------------------------------------

latent_size = 100

batch_size = 128
num_epochs = 64

# Models
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        hidden_size = 1000

        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, target_length_samples),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        hidden_size = 1000
        output_size = 1
        self.main = nn.Sequential(
            nn.Linear(target_length_samples, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
# ------------------------------------------------------------

class AudioDataSet(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.files = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.wav'):
                    self.files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Transform the audio to the right format.
        ## Make the audio mono
        waveform = self.ensure_mono(waveform)

        ## Make uniform audio length and sample rate.
        waveform = self.normalize_length(waveform, 
                                         sample_rate, 
                                         target_sample_rate, 
                                         target_length_samples
                                        )
        # ------------------------------------------------------------

        return waveform

    def ensure_mono(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.mean(0, keepdim=False)
        return waveform
    
    def normalize_length(self, input_tensor, sample_rate, target_sample_rate, target_length_samples):
        # Convert to correct sample rate
        if sample_rate != target_sample_rate:
            input_tensor = transforms.Resample(sample_rate, target_sample_rate)(input_tensor)
        
        # Convert to correct length
        current_size = len(input_tensor)
        if current_size < target_length_samples:
            # If the current size is smaller, pad with zeros.
            padding = (0, target_length_samples - current_size)
            output_tensor =  nn.functional.pad(input_tensor, padding)
        elif current_size > target_length_samples:
            # If the current size is larger, truncate the tensor.
            output_tensor = input_tensor[:target_length_samples]
        else:
            # If the current size matches the target size, no need to change.
            output_tensor = input_tensor
        return output_tensor

dataset = AudioDataSet(sys.argv[1])
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0
                                    )

generator = Generator()
discriminator = Discriminator()

optim_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002)
optim_generator = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()

# Training loop
for epoch in range(0, num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    for i, data in enumerate(loader, 0):
        discriminator_result = discriminator(data)
        
        # Generate fake batch.
        current_batch_size = len(data)
        latent_vector = torch.randn(current_batch_size, latent_size)
        generator_result = generator(latent_vector)
        # ------------------------------------------------------------

        label_for_fake = torch.zeros(current_batch_size, 1)
        label_for_real = torch.ones(current_batch_size, 1)
        
        error_real = loss(discriminator_result, label_for_real)
        error_real_logging_value = error_real.item()
        error_real.backward()

        error_fake = loss(discriminator(generator_result), label_for_fake)
        error_fake_logging_value = error_fake.item()
        error_fake.backward()

        error_discriminator = error_real + error_fake
        error_discriminator_logging_value = error_discriminator.item()
        optim_discriminator.step()

        logging_period = 2
        if i % logging_period == 0 or i == len(loader) - 1:
            print("- Batch {}/{}".format(i, len(loader)))
            print("-- Error real: " + str(error_real_logging_value))
            print("-- Error fake: " + str(error_fake_logging_value))
            print("-- Error combined: " + str(error_discriminator_logging_value))
        

