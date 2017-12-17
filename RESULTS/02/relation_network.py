import json
import numpy as np
import string
import tqdm

from keras import backend as K
from keras.layers import Layer, Input, Reshape, Embedding, LSTM, Dense, Lambda, Dropout
from keras.models import Model

# KERAS OPTIONS
K.set_image_data_format = "channels_last"


def relation_network(max_number_of_objects_in_scene=10, object_features_dim=13, max_question_length=43,
                     WORD_EMBEDDING_DIM=32, LSTM_UNITS=256,
                     G_FC1=512, G_FC2=512, G_FC3=512, G_FC4=512,
                     F_FC1=512, F_FC2=1024, F_DROPOUT2=0.02, answers_vocabulary_length=29):
    
    # scenes_input === (n, max_number_of_objects_in_scene, object_features_dim)
    # questions_input === (n, max_question_length)
    
    # Inputs
    scenes_input_tensor = Input(shape=(max_number_of_objects_in_scene, object_features_dim,))
    questions_input_tensor = Input(shape=(max_question_length,))
    
    # Process input for g_theta from [scenes_input, questions_input],
    # by making all object pairs for each scene and concatenating question in each pair
    # g_input === (n,
    #              max_number_of_objects_in_scene*max_number_of_objects_in_scene,
    #              2*object_features_dim+question_features_dim)
    g_input = process_scenes_and_questions(max_number_of_objects_in_scene=max_number_of_objects_in_scene,
                                           object_features_dim=object_features_dim,
                                           max_question_length=max_question_length,
                                           WORD_EMBEDDING_DIM=WORD_EMBEDDING_DIM,
                                           LSTM_UNITS=LSTM_UNITS)([scenes_input_tensor, questions_input_tensor])
    
    # G
    # Run each g_input through g_model to make g_output
    # g_model runs g_theta on each object_pair+question in each sample
    # g_output === (n, G_FC4)
    g_output = g_model(max_number_of_objects_in_scene=max_number_of_objects_in_scene,
                       object_features_dim=object_features_dim,
                       question_features_dim=LSTM_UNITS,
                       G_FC1=G_FC1, G_FC2=G_FC2, G_FC3=G_FC3, G_FC4=G_FC4)(g_input)
    
    # F
    # f_output === (n, answer_vocabulary_length)
    f_output = f_phi(input_dim=G_FC4,
                     F_FC1=F_FC1, F_FC2=F_FC2, F_DROPOUT2=F_DROPOUT2, F_FC3=answers_vocabulary_length)(g_output)
    
    relation_network = Model(inputs=[scenes_input_tensor, questions_input_tensor], outputs=[f_output])
    
    return relation_network


def process_scenes_and_questions(max_number_of_objects_in_scene=10,
                                 object_features_dim=13,
                                 max_question_length=43,
                                 WORD_EMBEDDING_DIM=32,
                                 LSTM_UNITS=256):
    '''
    scenes_input === n x max_number_of_objects_in_scene x object_features_dim
    '''
    # Inputs to model
    scenes_input = Input(shape=(max_number_of_objects_in_scene, object_features_dim,))
    questions_input = Input(shape=(max_question_length,))

    # Make question_features using Embedding+LSTM
    question_embeddings = Embedding(max_question_length, WORD_EMBEDDING_DIM, mask_zero=True)(questions_input)
    question_features = LSTM(LSTM_UNITS)(question_embeddings)

    # Make all object pairs and concatenate question_features to each pair 
    g_input = MakeGInput()([scenes_input, question_features])
    
    return Model(inputs=[scenes_input, questions_input], outputs=[g_input])


class MakeGInput(Layer):
    def __init__(self):
        super(MakeGInput, self).__init__()

    def build(self, input_shape):
        self.shape = input_shape
        super(MakeGInput, self).build(input_shape)

    def call(self, inputs, **kwargs):
        '''
        inputs[0] = scenes_input === (n, max_number_of_objects_in_scene, object_features_dim)
        inputs[1] = question_features === (n, questions_LSTM_dim)
        '''
        scenes_input = inputs[0]
        question_features = inputs[1]
        
        # Calc
        max_number_of_objects_in_scene = scenes_input.shape[1]
        
        # MAKE ALL OBJECT PAIRS, AND CONCATENATE QUESTION
        
        # Arrange one of the pair
        scenes_input_i = K.expand_dims(scenes_input, axis=1)
        
        # Repeat this pair in axis 1
        scenes_input_i1 = K.repeat_elements(scenes_input_i, rep=max_number_of_objects_in_scene, axis=1)

        # Reshape the second of the pair
        scenes_input_j = K.expand_dims(scenes_input, axis=2)

        # Arrange question_features to concatenate
        question_features1 = K.expand_dims(question_features, axis=1)
        question_features2 = K.repeat_elements(question_features1, rep=max_number_of_objects_in_scene, axis=1)
        question_features3 = K.expand_dims(question_features2, axis=2)

        # Concatenate question_features to second pair
        scenes_input_j_and_q = K.concatenate([scenes_input_j, question_features3], axis=-1)

        # Repeat second of the pair i axis 2
        scenes_input_j_and_q1 = K.repeat_elements(scenes_input_j_and_q, rep=max_number_of_objects_in_scene, axis=2)

        # Concatenate all
        g_input0 = K.concatenate([scenes_input_i1, scenes_input_j_and_q1], axis=-1)
        
        # Reshape to have all object pairs, i.e. 10*10=100 elements per scene-question sample
        g_input = K.reshape(g_input0,
                            (-1,
                             max_number_of_objects_in_scene*max_number_of_objects_in_scene,
                             g_input0.shape[-1].value))

        return g_input


def g_model(max_number_of_objects_in_scene=10, object_features_dim=13, question_features_dim=256,
            G_FC1=512, G_FC2=512, G_FC3=512, G_FC4=512):
    
    number_of_object_pairs = max_number_of_objects_in_scene * max_number_of_objects_in_scene
    features_dim = 2 * object_features_dim+question_features_dim
    
    g_input = Input(shape=(number_of_object_pairs, features_dim))
    
    # Convert g_input === (n, number_of_object_pairs, features_dim) into
    # g_theta_input === (n*number_of_object_pairs, features_dim) so that
    # g_theta can be applied on every object pair in every question+scene sample
    g_theta_input = Lambda(condense_batch_size_and_number_of_object_pairs)(g_input)
    
    # Apply g_theta
    g_theta_output = g_theta(features_dim=features_dim,
                             G_FC1=G_FC1, G_FC2=G_FC2, G_FC3=G_FC3, G_FC4=G_FC4)(g_theta_input)
    
    # Convert g_theta_output === (n*number_of_object_pairs, G_FC4) into
    # g_theta_outputs === (n, number_of_object_pairs, G_FC4) so that the
    # the g_theta outputs of all object pairs (in each question+scene sample) can be summed
    g_theta_outputs = Lambda(expand_batch_size_and_number_of_object_pairs,
                             arguments={'number_of_object_pairs': number_of_object_pairs})(g_theta_output)
    
    # Sum the g_theta outputs of all object pairs (in each question+scene sample)
    g_theta_outputs_sum = Lambda(sum_g_theta_outputs)(g_theta_outputs)
    
    return Model(inputs=[g_input], outputs=[g_theta_outputs_sum])


def g_theta(features_dim=282, G_FC1=512, G_FC2=512, G_FC3=512, G_FC4=512):
    g_theta_input = Input(shape=(features_dim,))
    x = Dense(G_FC1, activation='relu')(g_theta_input)
    x = Dense(G_FC2, activation='relu')(x)
    x = Dense(G_FC3, activation='relu')(x)
    g_theta_output = Dense(G_FC4, activation='relu')(x)
    return Model(inputs=[g_theta_input], outputs=[g_theta_output])


def condense_batch_size_and_number_of_object_pairs(x):
    return K.reshape(x, (-1, x.shape[-1].value))


def expand_batch_size_and_number_of_object_pairs(x, number_of_object_pairs=100):
    return K.reshape(x, (-1, number_of_object_pairs, x.shape[-1].value))


def sum_g_theta_outputs(g_theta_outputs):
    return K.sum(g_theta_outputs, axis=1)


def f_phi(input_dim=512, F_FC1=512, F_FC2=1024, F_DROPOUT2=0.02, F_FC3=29):
    f_input = Input(shape=(input_dim,))
    x = Dense(F_FC1, activation='relu')(f_input)
    x = Dense(F_FC2, activation='relu')(x)
    x = Dropout(F_DROPOUT2)(x)
    f_output = Dense(F_FC3, activation='softmax')(x)
    return Model(inputs=[f_input], outputs=[f_output])
