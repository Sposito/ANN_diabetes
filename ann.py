import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
import matplotlib.pyplot as plt

seed = 9

np.random.seed(seed)

def read_cvs_dataset(path, col_label):
    dataset = np.loadtxt(path, delimiter=',')
    print("Dataset format : " + dataset.shape)
    input_attributes = dataset[:,0:col_label]
    output_attributes = dataset[:, col_label]
    print('Formato das variáveis de entrada (input variables): ', input_attributes.shape)
    print('Formatoda classede saída(output variables): ',output_attributes.shape)
    #print(X[0])
    #print(Y[0])
    return(input_attributes,output_attributes)


def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    return model


def print_model (model,fich):
    from keras.utils import plot_model
    plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)


def compile_model (model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def fit_model ( model,input_attributes,output_attributes ):
    history = model.fit(input_attributes, output_attributes, validation_split=0.33, epochs=150, batch_size=10, verbose=2)
    return history

def print_history_accuracy (history):
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')    
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def model_evaluate(model,input_attributes,output_attributes):
    print("########### inicio do evaluate###############################\n" )
    scores = model.evaluate(input_attributes, output_attributes)
    print("\n metrica: %s: %.2f%%\n"% (model.metrics_names[1], scores[1]*100))


def model_print_predictions(model,input_attributes,output_attributes):
    previsoes = model.predict(input_attributes) 
    LP=[]
    #  arredondar para 0 ou 1 pois pretende - se um output binário
    for prev in previsoes:
        LP.append(round(prev[0]))
    #LP = [round(prev[0]) for prev in previsoes]
    for i in range( len(output_attributes)):
        print(" Class:",output_attributes[i]," previsão:",LP[i])
        if i>10: break

def ciclo_completo():
    (input_attributes,output_attributes) = read_cvs_dataset("pima-indians-diabetes.csv", 8)
    model = create_model()
    print_model(model,"model_MLP.png")
    compile_model (model)
    history= fit_model(model, input_attributes, output_attributes)
    print_history_accuracy(history)
    print_history_loss(history)
    model_evaluate(model, input_attributes, output_attributes)
    model_print_predictions(model, input_attributes, output_attributes)