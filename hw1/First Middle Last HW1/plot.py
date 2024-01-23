import matplotlib.pyplot as plt

# Data for epochs, training loss, training accuracy, validation margin, and validation accuracy
epochs = list(range(1, 26))
# training_loss = [1.799, 1.606, 1.530, 1.476, 1.434, 1.397, 1.364, 1.335, 1.309, 1.287, 
#                  1.263, 1.241, 1.221, 1.202, 1.184, 1.166, 1.148, 1.131, 1.116, 1.099, 
#                  1.084, 1.068, 1.052, 1.038, 1.023]
# training_accuracy = [0.373, 0.443, 0.471, 0.491, 0.506, 0.519, 0.534, 0.540, 0.551, 0.560, 
#                      0.568, 0.577, 0.585, 0.591, 0.599, 0.606, 0.609, 0.617, 0.621, 0.627, 
#                      0.633, 0.638, 0.645, 0.650, 0.655]
# validation_margin = [-0.402, -0.459, -0.456, -0.480, -0.491, -0.493, -0.507, -0.523, -0.538, 
#                      -0.528, -0.535, -0.570, -0.528, -0.538, -0.559, -0.562, -0.566, -0.562, 
#                      -0.582, -0.593, -0.588, -0.612, -0.599, -0.612, -0.616]
# validation_accuracy = [0.431, 0.460, 0.479, 0.484, 0.502, 0.507, 0.514, 0.520, 0.525, 0.531, 
#                        0.537, 0.536, 0.547, 0.547, 0.543, 0.549, 0.553, 0.555, 0.554, 0.556, 
#                        0.563, 0.557, 0.563, 0.568, 0.564]

training_loss = [1.833,1.634,1.558,1.506,1.465,1.431,1.402,1.374,1.352,1.329,1.309,1.289,1.273,1.256,1.241,1.227,1.211,1.199,1.186,1.173,1.161,1.148,1.138,1.128,1.114]
training_accuracy = [0.36,0.433,0.46,0.477,0.491,0.503,0.513,0.525,0.533,0.544,0.55,0.557,0.563,0.568,0.574,0.578,0.584,0.588,0.596,0.598,0.603,0.606,0.61,0.614,0.619]
validation_margin = [-0.397,-0.429,-0.478,-0.475,-0.502,-0.5,-0.509,-0.523,-0.507,-0.53,-0.536,-0.529,-0.537,-0.547,-0.571,-0.558,-0.577,-0.582,-0.571,-0.598,-0.566,-0.6,-0.595,-0.6,-0.6]
validation_accuracy = [0.421,0.451,0.467,0.478,0.49,0.498,0.503,0.509,0.516,0.524,0.527,0.526,0.534,0.536,0.536,0.539,0.541,0.545,0.542,0.545,0.544,0.541,0.55,0.549,0.552]


# Creating subplots
fig, axs = plt.subplots(2, figsize=(10, 8))

# Plotting validation loss and validation margin
axs[0].plot(epochs, validation_margin, label='Validation Margin', color='r', marker='o')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Validation Margin')
axs[0].set_title('Validation Margin per Epoch')
axs[0].grid(True)
axs[0].legend()

# Plotting validation accuracy
axs[1].plot(epochs, validation_accuracy, label='Validation Accuracy', color='b', marker='x')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Validation Accuracy')
axs[1].set_title('Validation Accuracy per Epoch')
axs[1].grid(True)
axs[1].legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

