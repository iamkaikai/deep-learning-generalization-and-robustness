import matplotlib.pyplot as plt

# Data for epochs, training loss, training accuracy, validation margin, and validation accuracy
epochs = list(range(1, 26))

#model 1
# validation_loss = [1.57, 1.482, 1.43, 1.398, 1.364, 1.346, 1.337, 1.321, 1.31, 1.298, 1.281, 1.287, 1.261, 1.262, 1.267, 1.263, 1.264, 1.269, 1.27, 1.282, 1.286, 1.275, 1.258, 1.294, 1.287]
# validation_margin = [-0.451, -0.497, -0.489, -0.511, -0.523, -0.541, -0.575, -0.579, -0.598, -0.584, -0.595, -0.616, -0.596, -0.606, -0.643, -0.647, -0.651, -0.655, -0.679, -0.693, -0.697, -0.712, -0.693, -0.747, -0.722]

#model 2
validation_loss = [1.598,1.508,1.457,1.426,1.399,1.381,1.363,1.345,1.343,1.32,1.321,1.321,1.302,1.307,1.301,1.292,1.318,1.287,1.301,1.297,1.298,1.316,1.297,1.295,1.305]
validation_margin = [-0.449, -0.475, -0.513, -0.518, -0.545, -0.561, -0.557, -0.578, -0.561, -0.592, -0.593, -0.603, -0.6, -0.613, -0.637, -0.617, -0.656, -0.66, -0.653, -0.684, -0.661, -0.682, -0.69, -0.682, -0.688]

# Creating subplots
fig, axs = plt.subplots(2, figsize=(10, 8))

axs[0].plot(epochs, validation_loss, label='Validation Loss', color='r', marker='o')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Validation Loss')
axs[0].set_title('Validation Loss per Epoch')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(epochs, validation_margin, label='Validation Margin', color='b', marker='x')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Validation Margin')
axs[1].set_title('Validation Margin per Epoch')
axs[1].grid(True)
axs[1].legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

