import glob
import os

from keras.optimizers import Adam

# RELATION NETWORK PARAMS
WORD_EMBEDDING_DIM = 32
LSTM_UNITS = 256
G_FC1, G_FC2, G_FC3, G_FC4 = 512, 512, 512, 512
F_FC1, F_FC2, F_DROPOUT2, F_FC3 = 512, 1024, 0.02, 29

# Train params
batch_size = 1024
epochs = 100
optimizer = Adam(lr=1e-7)
loss = 'categorical_crossentropy'

# Experiment params
prev_experiment_number = sorted([int(d.split('/')[-2]) for d in glob.glob(os.path.join("./RESULTS", "[0-9]*/"))])[-1]
experiment_number = prev_experiment_number + 1
print("Experiment number:", experiment_number)

# Make the dir if it doesn't exist
experiment_dir = "./RESULTS/{0:02d}".format(experiment_number)
if not os.path.exists(experiment_dir):
    print("Making dir", experiment_dir)
    os.makedirs(experiment_dir)

# Copy .py files to experiment_dir
os.system("cp *.py " + experiment_dir)
print("Copied *.py to", experiment_dir)

# FILE NAMES
# Scenes
state_description_train_file = '/home/voletiv/Datasets/CLEVR_v1.0/scenes/CLEVR_train_scenes.json'
state_description_val_file = '/home/voletiv/Datasets/CLEVR_v1.0/scenes/CLEVR_val_scenes.json'
# Questions
questions_train_file = '/home/voletiv/Datasets/CLEVR_v1.0/questions/CLEVR_train_questions.json'
questions_val_file = '/home/voletiv/Datasets/CLEVR_v1.0/questions/CLEVR_val_questions.json'
questions_test_file = '/home/voletiv/Datasets/CLEVR_v1.0/questions/CLEVR_test_questions.json'
