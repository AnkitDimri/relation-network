from keras.optimizers import Adam

# RELATION NETWORK PARAMS
WORD_EMBEDDING_DIM = 32
LSTM_UNITS = 256
G_FC1, G_FC2, G_FC3, G_FC4 = 512, 512, 512, 512
F_FC1, F_FC2, F_DROPOUT2, F_FC3 = 512, 1024, 0.02, 29

# Train params
batch_size = 1024
epochs = 100
optimizer = Adam(lr=1e-4)
loss = 'categorical_crossentropy'

# FILE NAMES
# Scenes
state_description_train_file = '/home/voletiv/Datasets/CLEVR_v1.0/scenes/CLEVR_train_scenes.json'
state_description_val_file = '/home/voletiv/Datasets/CLEVR_v1.0/scenes/CLEVR_val_scenes.json'
# Questions
questions_train_file = '/home/voletiv/Datasets/CLEVR_v1.0/questions/CLEVR_train_questions.json'
questions_val_file = '/home/voletiv/Datasets/CLEVR_v1.0/questions/CLEVR_val_questions.json'
questions_test_file = '/home/voletiv/Datasets/CLEVR_v1.0/questions/CLEVR_test_questions.json'
