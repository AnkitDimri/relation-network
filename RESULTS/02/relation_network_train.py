import numpy as np
import tensorflow as tf
import tqdm

from relation_network import *
from relation_network_train_params import *
from relation_network_train_functions import *

########################################################################
# LOAD STUFF
########################################################################

# Load questions and answers
question_answer_stuff = np.load("question_answer_stuff.npz")
max_question_length = question_answer_stuff["max_question_length"]
questions_image_indices_train = question_answer_stuff["questions_image_indices_train"]
questions_word_indices_train = question_answer_stuff["questions_word_indices_train"]
questions_image_indices_val = question_answer_stuff["questions_image_indices_val"]
questions_word_indices_val = question_answer_stuff["questions_word_indices_val"]
questions_image_indices_test = question_answer_stuff["questions_image_indices_test"]
questions_word_indices_test = question_answer_stuff["questions_word_indices_test"]
answers_vocabulary = question_answer_stuff["answers_vocabulary"]
one_hot_answers_train = question_answer_stuff["one_hot_answers_train"]
one_hot_answers_val = question_answer_stuff["one_hot_answers_val"]

# Load state_descriptions
state_description_matrix = np.load("state_description_matrix.npz")
max_number_of_objects_in_scene = state_description_matrix["max_number_of_objects_in_scene"]
object_features_dim = state_description_matrix["object_features_dim"]
state_description_matrix_train = state_description_matrix["state_description_matrix_train"]
state_description_matrix_val = state_description_matrix["state_description_matrix_val"]

########################################################################
# MAKE TRAINING, VALIDATION and TESTING BATCHES
########################################################################

# Train samples
# rn_input = [scenes_input_train, questions_input_train]
# scenes_input === n x max_number_of_objects_in_scene x object_features_dim
# questions_input === n x max_question_length
questions_input_train = questions_word_indices_train
scenes_input_train = np.zeros((len(questions_input_train),
                               state_description_matrix_train.shape[1],
                               state_description_matrix_train.shape[2]))
for i, image_index in enumerate(tqdm.tqdm(questions_image_indices_train)):
    scenes_input_train[i] = state_description_matrix_train[image_index]

# Val samples
questions_input_val = questions_word_indices_val
scenes_input_val = np.zeros((len(questions_input_val),
                               state_description_matrix_val.shape[1],
                               state_description_matrix_val.shape[2]))
for i, image_index in enumerate(tqdm.tqdm(questions_image_indices_val)):
    scenes_input_val[i] = state_description_matrix_val[image_index]

# Test samples
questions_input_test = questions_word_indices_test
# scenes_input_test = np.zeros((len(questions_input_test),
#                                state_description_matrix_test.shape[1],
#                                state_description_matrix_test.shape[2]))
# for i, image_index in enumerate(tqdm.tqdm(questions_image_indices_test)):
#     scenes_input_test[i] = state_description_matrix_test[image_index]

########################################################################
# MAKE AND COMPILE RELATION NETWORK
########################################################################

np.random.seed(29)
tf.set_random_seed(29)

# Make the relation network
rn = relation_network(max_number_of_objects_in_scene=max_number_of_objects_in_scene,
                      object_features_dim=object_features_dim,
                      max_question_length=max_question_length,
                      answers_vocabulary_length=len(answers_vocabulary),
                      WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIM, LSTM_UNITS=LSTM_UNITS,
                      G_FC1=G_FC1, G_FC2=G_FC2, G_FC3=G_FC3, G_FC4=G_FC4,
                      F_FC1=F_FC1, F_FC2=F_FC2, F_DROPOUT2=F_DROPOUT2)

# Compile
rn.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

########################################################################
# TRAIN
########################################################################

checkpointAndMakePlots = CheckpointAndMakePlots(save_dir=experiment_dir)

# Fit
rn.fit(x=[scenes_input_train, questions_input_train], y=[one_hot_answers_train],
       batch_size=batch_size, epochs=epochs, callbacks=[checkpointAndMakePlots],
       validation_data=([scenes_input_val, questions_input_val], one_hot_answers_val),
       shuffle=True, class_weight=None)

########################################################################
# TEST
########################################################################

# # Test
# rn_preds_test = rn.predict([scenes_input_train, questions_input_test],
#                            batch_size=batch_size, verbose=True)

# # Write the predictions into txt file
# with open("rn_pred_test.txt", 'w') as f:
#     for softmax_pred in rn_preds_test:
#         f.write(answers_vocabulary[np.argmax(softmax_pred)] + '\n')

