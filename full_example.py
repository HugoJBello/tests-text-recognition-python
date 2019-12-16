# https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6

# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import json
nltk.download('punkt')

stemmer = LancasterStemmer()

training_data = []

with open('data/examples_violence.json') as json_file:
    data = json.load(json_file)
    for text in data:
        training_data.append({"class":"violencia machista", "sentence":text})
        print(training_data)

with open('data/examples_no_violence.json') as json_file:
    data = json.load(json_file)
    for text in data:
        training_data.append({"class":"no violencia machista", "sentence":text})
        print(training_data)


# tokeninzation 
words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)







# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w])
print (training[i])
print (output[i])



import numpy as np
import time

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))






def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
            'datetime': now.strftime("%Y-%m-%d %H:%M"),
            'words': words,
            'classes': classes
            }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)



X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")



test_new_no_violence="La encomiable resistencia de Denis Shapovalov se transforma en una masa viscosa de la que no es fácil despegarse, pero el desagradable efecto se agota y llega el gran momento cuando el joven estrella una derecha en la red: 6-3 y 7-6(7), en 1h 55m. Entonces, Rafael Nadal cae a plomo al suelo y la Caja Mágica entra en erupción. Ocho años después, España reconquista la Copa Davis y eleva la sexta ensaladera de su historia, porque antes Roberto Bautista ha logrado una emotiva victoria en la apertura de la final contra Canadá. Se añade el festejo a los de 2000, 2004, 2008, 2009 y 2011, y Suecia queda ahora a un solo trofeo en el historial.  MÁS INFORMACIÓN España conquista la Davis de NadalCOLUMNA | 'El abrazo a Bautista', por TONI NADAL España conquista la Davis de NadalFOTOGALERÍA Las escenas de la final España conquista la Davis de NadalEmoción y trasnoche para un giro total  Del Sant Jordi a la Caja Mágica, de aquel éxito de hace 19 años a este último en 2019, hay infinidad de capítulos reservados, nombres propios y escenas, pero existe un eje vertebrador llamado Nadal. No estuvo presente el balear en la primera ensaladera, porque entonces aún estaba desarrollándose el chico que se convertiría en mito y que ese día fue el abanderado, mientras el fotograma final correspondía a Juan Carlos Ferrero; sin embargo, el hilo conductor del éxito en las dos últimas décadas tiene origen en Manacor. Preside él, imponente, el último laurel, el primero en el rompedor diseño propuesto por Gerard Piqué y su equipo. Se cacareaba desde hace meses el cansino sobrenombre de la Davis de Piqué, y al final la realidad del torneo ha derivado en otro: la Davis de Nadal.  A contrapié, la majestuosa actuación del mallorquín en la última semana –ocho puntos de los once obtenidos– contradice de alguna manera el espíritu de fondo de la Copa Davis, competición que acostumbra a premiar la coralidad. Pero esta vez, más nunca, ha sido un triunfo de autor. Ha ido España casi siempre a remolque y él ha ido empujando la nave un día sí y otro también. De principio a fin, desdoblándose. Resolvió los cruces contra Rusia y Argentina, despachó a Croacia y sentenció a Gran Bretaña en el territorio de dobles que tan bien manejan los británicos. Y en la final, más Nadal, presente en cuatro de las cinco rúbricas de las ensaladeras previas, aunque no compitió en la final de Mar del Plata (2008).  Debutó con 17 años en Brno, cuando Jordi Arrese le dio la alternativa para medirse al checo Jiri Novak, y ese mismo año ya dejó huella en la competición al derrotar en la final de La Cartuja al estadounidense Andy Roddick siendo todavía un rookie. A partir de ahí, un largo vuelo que comprende 23 eliminatorias, con un saldo final de 29 victorias y una sola derrota en los individuales, y 8-4 en dobles. Ha vivido España a su compás, en función de que le respetase su cuerpo y su calendario le permitiera aumentar en mayor medida la implicación; también sujeta a la circunstancia institucional, porque la década gloriosa dio paso luego a un periodo sombrío (en ocasiones esperpéntico) de inestabilidad. ADVERTISING  inRead invented by Teads   Contra una torre de Babel Corren ahora buenos tiempos, la federación intenta relanzar la base y el presente sonríe con la inauguración triunfal del nuevo formato. Un éxito que llevará para siempre el sello de Nadal, respaldado por el otro protagonista, Bautista, autor de dos puntos; el restante fue obra de Feliciano López y Marcel Granollers. Aunque se contuviera por dentro, en ese brinco del castellonense hacia el banquillo viajaban un millón de emociones. Proclive a relatos épicos y heroicidades, la Davis regaló este domingo otro de esos capítulos inolvidables, porque la competición le había reservado un emotivo episodio en la jornada final.  Después de haber perdido a su padre hace solo tres días y de haber abandonado la concentración tras jugar dos partidos –derrota en la apertura, contra Andrey Rublev, y triunfo ante el croata Nikola Mektic el segundo día–, Bautista cogió el coche y regresó a Madrid el sábado, y al día siguiente quiso saltar a la pista. Sergi Bruguera le alineó, en su condición de número dos del equipo, y firmó el primer punto de la final para España al vencer a Auger-Aliassime por 7-6(3) y 6-3, en 1h 38m.  Pese a la desgracia familiar, el número nueve del mundo hizo toda una demostración de aplomo, decidiendo jugar y abrir la serie frente a la multiétnica Canadá; esta, una verdadera torre de Babel. Sus componentes mezclan orígenees ruso, israelí (Shapovalov), checo (Pospisil) y togolés, en el caso de Aliassime. El joven, de 19 años, es uno de los tenistas a los que se observa con lupa, porque tiene muchos ingredientes para hacerse notar en los próximos años. Este año alcanzó las semifinales de Miami y hace cuatro se proclamó campeón júnior de la Davis, precisamente en la Caja Mágica.  Una atmósfera emotiva Aquejado de un esguince de tobillo, todavía no había intervenido en la competición, pero su técnico apostó por él y no le salió la jugada. Sí, en cambio, a Bruguera. Introdujo a Bautista y este logró una de las victorias más importantes de su carrera. En una atmósfera emotiva, el castellonense (31 años) sacó del maletín su buen oficio y desarmó al rival, que cedió en su apuesta a todo o nada, con 45 errores no forzados. Le pesó al canadiense la pérdida del tie break en el primer parcial, y en el segundo Bautista administró con madurez las distancias. Quebró para 3-0 y pese a la réplica (3-2) cerró con autoridad el duelo.  Entonces alzó los brazos, dedicó el punto a su padre y después del abrazo con el capitán se abalanzó hacia el resto del equipo. A base de corazón, situó a España a un solo punto de la sexta ensaladera y a continuación remató el trabajo el de casi siempre, Nadal, artífice fundamental del reencuentro con la felicidad. A la tercera bola de partido y después de levantar una de set en contra, el número uno decidió. Suma 29 victorias y una sola derrota en los partidos individuales, y ostenta además el récord absoluto de 31 triunfos consecutivos incluyendo los dobles. Redondea así una temporada fabulosa, con dos grandes (Roland Garros y el US Open), otros dos títulos (Roma y Montreal) y el número uno.  Y estaba escrito, con todas las letras: esta Davis era de él.  LAS SEIS ENSALADERAS DE ESPAÑA 2000. España, 3 - Australia, 1. Palau Sant Jordi (Barcelona). Costa, Ferrero, Corretja, Balcells.  2004. Estados Unidos, 2 - España, 3. Estadio La Cartuja (Sevilla). Moyá, Nadal, Ferrero, Robredo.  2008. Argentina, 1 - España, 3. Mar del Plata (Argentina). Nadal, Ferrer, Feliciano López, Verdasco.  2009. República Checa, 0 - España, 5. Palau Sant Jordi (Barcelona). Nadal, Ferrer, Feliciano López, Verdasco.  2011. Argentina, 1 - España, 3. Estadio La Cartuja (Sevilla). Nadal, Ferrer, Feliciano López, Verdasco.  2019. España, 2 - Canadá, 0. Caja Mágica (Madrid). Nadal, Bautista, Carreño, Feliciano López, Granollers."
test_new_violence="Una activista mexicana defensora de los indígenas ha sido asesinada este jueves en el Estado mexicano de Sonora. Raquel Padilla Ramos, historiadora y antropóloga murió debido a las “heridas de arma blanca” que le infligió su pareja, un hombre de 55 años, en una casa en un pequeño poblado llamado Ures, según ha anunciado la Fiscalía estatal. La policía local detuvo al agresor en el mismo momento en que cometía el crimen, sobre las cuatro de la tarde. La Procuraduría local ha abierto una investigación por feminicidio contra el hombre. La víctima, de 53 años, trabajaba como investigadora para el Instituto Nacional de Antropología e Historia mexicano (INAH).  MÁS INFORMACIÓN Asesinada por su pareja una defensora de los indígenas en el Estado mexicano de SonoraLiberados los dos activistas indígenas secuestrados en Guerrero La violencia provoca casi nueve millones de desplazados en México desde 2011 México pone en marcha un programa para buscar a 40.000 desaparecidos  De perfil académico y feminista, las redes sociales de Padilla Ramos se han vuelto este viernes una especie de bitácora rememorativa. En el muro de Facebook la activista aún se condolía por el asesinato de una mujer el pasado octubre en Playa del Carmen, en el Estado de Quintana Roo: “Tenía 25 años y le robaron la vida”, escribió. “A usted también se la robaron, descanse en paz”, lamentan ahora los que conocieron a Ramos Padilla. Entremezcladas con estas publicaciones, la activista también compartía con regularidad fotos y anécdotas junto a sus hijos y su pareja.  Días antes a su muerte, la activista había dejado numerosas reflexiones en sus redes sociales sobre la matanza en Bavispe, en la que el pasado lunes seis niños y tres mujeres de la familia LeBarón fueron brutalmente asesinados. “La masacre de mujeres y niños es lo importante”, compartió en su cuenta de Twitter el comentario de la periodista mexicana Lydia Cacho. Desde que trascendió la noticia en este poblado a 250 kilómetros al norte de Ures, la académica había ofrecido varias explicaciones sobre la historia del lugar. “Bavispe fue un pueblo de los indios ópatas, refundado como pueblo de misión por la Compañía de Jesús”, había señalado.  Padilla Ramos tenía un importante recorrido en el mundo académico. Había cursado una licenciatura en Ciencias Antropológicas en la Universidad Autónoma de Yucatán, una maestría en Ciencias Antropológicas y un doctorado en Estudios Mesoamericanos en la Universidad de Hamburgo, en Alemania. El secretario de Cultura de Ciudad de México, Alfonso Suárez, ha sido uno de los primeros en sumarse a las condolencias este viernes. “Condenamos la violencia y exigimos justicia”, ha escrito en su cuenta de Twitter. El INAH también lamentó  la sensible pérdida por el fallecimiento  de la “destacada escritora e intelectual”, a quién calificó como “una luchadora incansable por los derechos territoriales y culturales de los pueblos indígenas”. El comunicado, sin embargo, levantó críticas por referirse al hecho como fallecimiento y no como feminicidio.  Sonora es uno de los Estados mexicanos con peores cifras en agresiones machistas en México. Solo entre enero y julio de este año, 24 mujeres fueron asesinadas por sus parejas en esa entidad. Unos números que lo convierten en el quinto más violento para las mujeres. El detenido por el homicidio de Ramos Padilla fue señalado por varios testigos como el autor del crimen y enfrentará un proceso por feminicidio."
classes=["violencia machista", "no violencia machista"]

# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results

classify(test_new_violence)
classify(test_new_no_violence)

print()
classify(test_new_violence, show_details=True)