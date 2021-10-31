#ESTE CODIGO DEBIERA EJECUTARSE INGRESANDO LOS 3 ARGUMENTOS CREADOS
#python TemasProyecto1.py --dir pet_images/ --arch vgg --dogfile dognames.txt

#-------------------------------------------------------------------------------------------------------------------------------
#TODO 0: RUNTIME

    #Capturar el tiempo de inicio del programa y luego el final, para restarlos y calcular el tiempo de runtime
    #La funcion sleep nos sirve para probar esto, pues duerme la ejecución los segundos que le digamos

from time import time, sleep

start_time = time()
print (start_time)

sleep(3)

end_time= time()
print (end_time)

tot_time = end_time - start_time
print ("Runtime en segundo" , tot_time)

#En formato HH:MM:SS

print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

    #hours = int( (tot_time / 3600) )
    #minutes = int( ( (tot_time % 3600) / 60 ) )
    #seconds = int( ( (tot_time % 3600) % 60 ) )

#--------------------------------------------------------------------------------------------------------------------------------
#TODO 1:ARGUMENTOS DE INPUT, EL USUARIO LO INGRESA AL EJECTURA EL CODIGO, PERO SI NO LOS INGRESARA CON ESTA FUNCION NOS ASEGURAMOS DE QUE HAYAN DEFAULT
#TemasProyecto1.py --dir pet_images/ --arch vgg --dogfile dognames.txt
import argparse

#CREAR UN ARRAY DE INPUT NECESARIOS
def get_input_args():

    # SE CREA UN OBJETO PARA LOS ARGUMENTOS
    parser = argparse.ArgumentParser()

    ##AGREGAMOS EL ARGUMENTO 1: PET_IMAGES/
    parser.add_argument('--dir', type = str, default = 'pet_images/',  help = 'Ruta de la carpeta de imagenes de perros')
    ##AGREGAMOS EL ARGUMENTO 2: ARQUITECTURA DEL MODELO CNN
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'Tipo modelo CNN (VGG, AlexNet , ResNet)' )
    ##AGREGAMOS EL ARGUMENTO 3: ARCHIVOS NOMBRE DE LOS PERRITOS
    parser.add_argument('--dogfile', type = str, default = 'dognames.txt', help = 'Ruta archivo txt con los nombres de perros' )

    # METEMOS TODO EL PARSE_ARGS() EN UNA VARIABLE
    in_args = parser.parse_args()

    # VEMOS SI ESTA BIEN DEFINIDOS LOS ARGUMENTOS
    print("Argument 1:", in_args.dir)
    print("Argument 2:", in_args.arch)
    print("Argument 3:", in_args.dogfile)

    #SE REEMPLAZA LA SALIDA DE LA FUNICON QUE ESTABA EN "NONE" A LOS ARGUMENTOS QUE HICIMOS
    return parser.parse_args()

#ESTA FUNCION CHECKEA QUE HICIMOS BIEN LA FUNCION DE LOS ARGUMENTOS ANTERIORES
def check_command_line_arguments(in_arg):

    if in_arg is None:
        print("CHEQUEO - Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("CHEQUEO - Command Line Arguments:\n     dir =", in_arg.dir,
              "\n    arch =", in_arg.arch, "\n dogfile =", in_arg.dogfile)

#OCUPAMOS LAS DOS FUNCIONES ANTERIORES
in_arg = get_input_args()
check_command_line_arguments(in_arg)

#--------------------------------------------------------------------------------------------------------------------------------
#TODO 2:ACA DEBEMOS CREAR UN DICCIONARIO CON LOS 40 TIPOS DE PERRO CREANDOLOS A PARTIR DEL NOMBRE DE LAS IMAGENES EN LA CARPETA DE IMAGENES DE PERRITOS
#ESE DICCIONARIO DEBE TENER LOS LABELS DEL PERRITO QUE SERIA EL MISMO NOMBRE DEL ARCHIVO SIN NUMEROS, SIN _ Y SIN MAYUSCULAS

from os import listdir

#PRIMERO NOS TRAERMOS LA LISTA DE TODAS LAS IMAGENES EN EL ARCHIVO DE PERRITOS (NOTAR QUE OCUPAMOS EL ARGUMENTO QUE DEFINIMOS EN EL PASO ANTERIOR PARA DECIR EL DIRECTORIO)
#filename_list = listdir("pet_images/")
filename_list = listdir(in_arg.dir)

# PROBAMOS QUE ESTA FUNCIONANDO TRAYENDONOS LOS PRIMEROS 5
print("\nCHEQUEO SI FUNCIONA EL DIRECTORIO - IMPRIME 10 SE LOS NOMBRES DE LOS ARCHIVOS EN LA CARPETA DE IMAGENES DE PERRITO:")
for idx in range(0, 5, 1):
    print("{:2d} file: {:>25}".format(idx + 1, filename_list[idx]) )

#CREAMOS LA PRIMERA FUNCION QUE CONVIERTE EL NOMBRE DE UN ARCHIVO DE IMAGEN EN EL DIRECTORIO QUE DIJIMOS A UN LABEL QUE QUEREMOS,
#QUE EN EL FONDO ES AL NOMBRE DEL ARCHIVO QUITARLE GUIONES, NUMEROS Y PASARLO A MINUSCULA
def creadorLabelTipoDog(nombreImagenDog):
    # Sets pet_image variable to a filename
    pet_image = nombreImagenDog

    # Sets string to lower case letters
    low_pet_image = pet_image.lower()

    # Splits lower case string by _ to break into words
    word_list_pet_image = low_pet_image.split("_")

    # Create pet_name starting as empty string
    pet_name = ""

    # Loops to check if word in pet name is only
    # alphabetic characters - if true append word
    # to pet_name separated by trailing space
    for word in word_list_pet_image:
        if word.isalpha():
            pet_name += word + " "

    # Strip off starting/trailing whitespace characters
    pet_name = pet_name.strip()

    return pet_name

    """# IMPRIMIMOS LOS LABELS CREADOS PARA VER EL RETURN
    print("\nFilename=", pet_image, "   Label=", pet_name) """

"""#LA PROBAMOS EN UN EJEMPLO
creadorLabelTipoDog("Boston_terrier_02259.jpg")

#AHORA LA PROBAMOS EN TODA LA LISTA DE ARCHIVOS
for item in range(0, len(filename_list), 1):
    creadorLabelTipoDog(filename_list[item])
"""

#AHORA DEBEMOS CREAR EL DICCIONARIO CON TODOS LOS LABEL USANDO LA FUNCION ANTERIOR EN LAS 40 IMAGENES Y METIENDO LOS LABELS AL DICCIONARIO
#PARA ESO CREAMOS UNA NUEVA FUNCION QUE VA A BUSCAR A UN DIRECTORIO TODAS LA IMAGENES, LES CREA SU LABEL Y LOS METE AL DICCIONARIO REVISANDO QUE NO SE REPITAN
def get_pet_labels(image_dir):

    #CREAMOS UN NUEVO DICCIONARIO, RECORDAR QUE UN DICCIONARIO TIENE ESTE FORMATO [KEY O ID]: [0] [1] [2]
    results_dic = dict()

    #VEMOS CUANTOS ARCHIVOS HAY EN EL DICCIONARIO EN UN PRINCIPIO
    items_in_dic = len(results_dic)
    print("\nELEMENTOS EN EL DICCIONARIO - N ITEMS AL PRINCIPIO=", items_in_dic)

    # AHORA DEBEMOS ADHERIR LOS LABEL (SOLO LOS LABEL) AL DICCIONARIO, OJO DEBE SER UN VALOR UNICO ASI QUE DEBEMOS HACER LA LÓGICA DE QUE SE AGREGUE SOLO SI NO EXISTE YA
    #CREAMOS LA LISTA DE LA IMAGENES DEL DIRECTORIO QUE INGRESA A LA FUNCION
    filenames = listdir(image_dir)

    #CREAMOS LA LISTA DE LOS LABELS Y LA LLENAMOS USANDO LA FUNCION QUE CREA LABELS, LA USAMOS SOBRE LA LISTA DE IMAGENES QUE CREAMOS RECIEN
    pet_labels = []
    for item in range(0, len(filenames), 1):
        pet_labels.append(creadorLabelTipoDog(filenames[item]))

    #CON ESTO YA CREAMOS LA LISTA DE LABELS QUE DEBEMOS SUMAR A LA DICCIONARIO, REVISAMOS QUE NO SE ENCUENTREN YA EN EL DICCIONARIO Y LOS VAMOS AGREGANDO

    for idx in range(0, len(filenames), 1):
        if filenames[idx] not in results_dic and filenames[idx][0] != ".":
             results_dic[filenames[idx]] = [pet_labels[idx]]
        else:
             print("** Warning: Key=", filenames[idx],
                   "already exists in results_dic with value =",
                   results_dic[filenames[idx]])

    #IMPRIMIMOS TODO EL DICCIONARIO PARA VER QUE FUNCIONO Y REVISAMOS CUANDO ELEMENTOS AGREGAMOS

    print("\nTODOS LOS KEY-VALOR AGREGADOS AL DICCIONARIO SON:")
    for key in results_dic:
        print("Filename=", key, "   Pet Label=", results_dic[key][0])

    items_in_dic = len(results_dic)
    print("\nELEMENTOS EN EL DICCIONARIO - N ITEMS DESPUES DE LA EJECUCION=", items_in_dic)

    return results_dic

"""#PROBAMOS LA FUNCION
get_pet_labels("pet_images/")"""

#FUNCION DE CHEQUEO DE AL ANTERIOR
def check_creating_pet_image_labels(results_dic):
    """    For Lab: Classifying Images - 9/10. Creating Pet Image Labels
    Prints first 10 key-value pairs and makes sure there are 40 key-value
    pairs in your results_dic dictionary. Assumes you defined the results_dic
    dictionary as was outlined in
    '9/10. Creating Pet Image Labels'
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
    Returns:
     Nothing - just prints to console
    """
    if results_dic is None:
        print("CHEQUEO - Doesn't Check the Results Dictionary because 'get_pet_labels' hasn't been defined.")
    else:
        # Code to print 10 key-value pairs (or fewer if less than 10 images)
        # & makes sure there are 40 pairs, one for each file in pet_images/
        stop_point = len(results_dic)
        if stop_point > 10:
            stop_point = 10
        print("\nCHEQUEO - Pet Image Label Dictionary has", len(results_dic),
              "key-value pairs.\nBelow are", stop_point, "of them:")

        # counter - to count how many labels have been printed
        n = 0

        # for loop to iterate through the dictionary
        for key in results_dic:

            # prints only first 10 labels
            if n < stop_point:
                print("{:2d} key: {:>30}  label: {:>26}".format(n+1, key,
                      results_dic[key][0]))

                # Increments counter
                n += 1

            # If past first 10 (or fewer) labels the breaks out of loop
            else:
                break

#CHEQUEAMOS ENTONCES, OJO QUE OCUPAMOS COMO DIRECTORIO EL ARGUMENTO DE DIRECTORIO QUE CREAMOS EN LE PRIMER PASO
results_dic = get_pet_labels(in_arg.dir)
check_creating_pet_image_labels(results_dic)


#--------------------------------------------------------------------------------------------------------------------------------

#TODO 4.1: CLASIFICADOR DE IMAGENES PRUEBA

#PRIMERO IMPORTO Y ME TRAIGO TODO EL CLASIFICADOR DE IMAGENES QUE VIENE YA HECHO DESDE LA WEB
#OJO PARA INSTALAR PIL Y PODER IMPORTARLO TUVE QUE CORRER EN POWERSHELL DE ANACONDA Y EJECUTAR "pip install Pillow==6.1"
import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

#ESTA ES LA FUNCION DEL CLASIFICADOR, LE PASO UNA IMAGEN Y EL MODELO QUE VOY A OCUPAR Y ME DEVUELVE QUE ES LA IMAGEN
def classifier(img_path, model_name):
    # load the image
    img_pil = Image.open(img_path)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # preprocess the image
    img_tensor = preprocess(img_pil)

    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)

    # wrap input in variable, wrap input in variable - no longer needed for
    # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
    pytorch_ver = __version__.split('.')

    # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the
    # affect of setting volatile = True (because we are using pretrained models
    # for inference) we can set requires_gradient to False. Here we just set
    # requires_grad_ to False on our tensor
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)

    # pytorch versions less than 0.4 - uses Variable because not-depreciated
    else:
        # apply model to input
        # wrap input in variable
        data = Variable(img_tensor, volatile = True)

    # apply model to input
    model = models[model_name]

    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.eval()

    # apply data to model - adjusted based upon version to account for
    # operating on a Tensor for version 0.4 & higher.
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        output = model(img_tensor)

    # pytorch versions less than 0.4
    else:
        # apply data to model
        output = model(data)

    # return index corresponding to predicted class
    pred_idx = output.data.numpy().argmax()

    return imagenet_classes_dict[pred_idx]

"""#LO PROBAMOS DE EJEMPLO (ACA NO LO IMPORTO PORQUE ESTAMOS EN EL MISMO ARCHIVO)
from classifier import classifier"""

# DEFINO LA IMAGEN
test_image="pet_images/Collie_03797.jpg"

# DEFINO EL TIPO DE MODELO DE CLASIFICACION QUE VOY A USAR
#      'vgg', 'alexnet', 'resnet'
model = "vgg"

# METO EL RESULTADO EN UN STRING
image_classification = classifier(test_image, model)

#IMPRIMIMOS EL RESULTADO
print("\nEL RESULTADO DE PRUEBA DEL CLASIFICADOR DE IMAGEN SOBRE LA IMAGEN", test_image, "USANDO EL MODELO:",
      model, "ES", image_classification)

#-----------------------------------------------------------------------------------------------------------------------------
#TODO 4.2: USAMOS EL CLASIFICADOR DE IMAGENES Y COMPARAMOS LOS RESULTADOS CON NUESTRO LABEL PARA VER SU PRECISION (ACA NO LO IMPORTO PORQUE ESTA ACA)
print("EL ACTUAL DICCIONARIO ES")
print(results_dic)
#RECORDAR ENTONCES QUE LA KEY DEL DICCIONARIO ES EL NOMBRE DEL ARCHIVO, Y EL PRIMERO CAMPO QUE LO ACOMPAÑA ES EL LABEL results_dic[key][0]
print("RECORDAR QUE EL DICCIONARIO SE ESTRUCTURA COMO")
for key in results_dic:
    print("Key es el Filename =", key , "y Valor [0] es el Pet Label=", results_dic[key][0])

#ENTONCES AHORA PARA OCUPAR EL CLASIFICADOR EN CADA ARCHIVO USAMOS LA KEY DEL DICCIONARIO
print("AHORA OCUPAMOS EL CLASIFICADOR SOBRE EL DICCIONARIO")
"""for key in results_dic:
    salida_modelo_sobre_diccionario = classifier("pet_images/" + key, "vgg")
    #print("Archivo: " , key , "Clasificado dice: " , salida_modelo_sobre_diccionario)

    salida_limpia = salida_modelo_sobre_diccionario.lower().strip()
    #print("Archivo: " , key , "Clasificado limpio: " , salida_limpia)

    #AGREGAMOS LA SALIDA DEL CLASIFICADOR AL DICCIONARIO PARA QUE QUEDE "KEY: LABEL DOG , SALIDA CLASIFICADOR"
    results_dic[key].append(salida_limpia)

    #AHORA COMPARAREMOS EL LABEL "results_dic[key][0]" CON LA SALIDA DEL CLASIFICADOR "results_dic[key][1]"

    if results_dic[key][0] in results_dic[key][1]:
        clasificador_correcto = 1
    else :
        clasificador_correcto = 0

    #AGREGAMOS ESE RESULTADO AL DICCIONARIO PARA QUE QUEDE ENTONCES "KEY: LABEL DOG , SALIDA CLASIFICADOR, COMPARACION"
    results_dic[key].append(clasificador_correcto)

print(results_dic)"""

#METEMOS TODO LO ANTERIOR EN UNA FUNCION Y LO PROBAMOS, ENTONCES ESTA FUNCION TOMA EL CLASIFICADOR Y LO USA SOBRE LAS IMAGENES EN LA RUTA Y EN EL
#DICCIONARIO, LOS COMPARA CON EL LABEL Y LOS AGREGA AL DICCIONARIO

"""from classifier import classifier"""
#usando los input argumentales de todo classify_images(in_arg.dir, results_dic, in_arg.arch)
#usando los valores classify_images("pet_images/", results_dic, "vgg"")
def classify_images(images_dir, results_dic, model):

    for key in results_dic:
       model_label = classifier(images_dir + key, model)
       salida_limpia = model_label.lower().strip()
       #print("Archivo: " , key , "Clasificado limpio: " , salida_limpia)

       #AGREGAMOS LA SALIDA DEL CLASIFICADOR AL DICCIONARIO PARA QUE QUEDE "KEY: LABEL DOG , SALIDA CLASIFICADOR"
       results_dic[key].append(salida_limpia)

       #AHORA COMPARAREMOS EL LABEL "results_dic[key][0]" CON LA SALIDA DEL CLASIFICADOR "results_dic[key][1]"

       if results_dic[key][0] in results_dic[key][1]:
           clasificador_correcto = 1
       else :
           clasificador_correcto = 0

       #AGREGAMOS ESE RESULTADO AL DICCIONARIO PARA QUE QUEDE ENTONCES "KEY: LABEL DOG , SALIDA CLASIFICADOR, COMPARACION"
       results_dic[key].append(clasificador_correcto)

#PROBAMOS
classify_images("pet_images/", results_dic, "vgg")
print(results_dic)


#------------------------------------------------------------------------------------------------------------------------
#TO DO 5: DEBEMOS CREAR OTRO NUEVO DICCIONARIO CON EL ARCHIVO DE "DOGNAMES.TXT" QUE TIENE TODOS LOS PERROS DEL MUNDO. lA IDEA ES CON ESE DICCIONARIO
#COMPARAR NUESTROS LABEL Y NUESTRA SALIDA DE CLASIFICADOR Y VER SI SON O NO PERROS, Y AGREGAR ESO COMO 2 CAMPOS MAS A NUESTRO DICCIONARIO ORIGINAL
"""
#CREAMOSUN NUEVO DICCIONARIO, OJO LO HACEMOS CON DICCIONARIO PORQUE ES MAS RAPIDO EL LOOKUP QUE UN SOLO ARRAY
diccionario_perros_txt = dict()

#ABRIMOS EL ARCHIVO TXT
txt_dogs = open("dognames.txt", "r")
#print (doc.read())
#print (doc.readline())

#READLINE() lee una a una las lineas del archivo, entonces lo recorremos hasta que este este vacio y lo metemos a una array primero
txt_dogs_lista = []
while txt_dogs.readline() != "":
    txt_dogs_lista.append(txt_dogs.readline().rstrip())
    #rstrip quita los /n que salen por defecto cuando hay saltos de paginas

#AHORA RECORREMOS TODO EL ARRAY ANTERIOR Y INGRESANDOLO AL NUEVO DICCIONARIO, SIEMPRE Y CUANDO NO ESTE
for i in range(0, len(txt_dogs_lista), 1):
    if txt_dogs_lista[i] not in diccionario_perros_txt:
        #ACA ES DONDE LE DECIMOS, SI NO ESTA EN EL DICCIONARIO, EN LA KEY CON NOMBRE DE LA LINEA DEL ARCHIVO INGRESA UN NUEVO CAMPO CON VALOR 1, ASI QUEDA
        #KEY = LINEA TXT; VALOR = 1
         diccionario_perros_txt[txt_dogs_lista[i]] = [1]

print("DICCIONARIO NUEVO DEL TXT")
print(diccionario_perros_txt)

for key in results_dic:

    if results_dic[key][0] in diccionario_perros_txt:
        label_en_txt = 1
    else: label_en_txt = 0

    results_dic[key].append(label_en_txt)

    if results_dic[key][1] in diccionario_perros_txt:
        clasificador_en_txt = 1
    else: clasificador_en_txt = 0

    results_dic[key].append(clasificador_en_txt)

print("ULTIMO DICCIONARIO CON VALIDACION SI LABEL Y CLASIFICADOR ESTAN EN DICCIONARIO DE TXT")
print(results_dic) """

"""QUEDO ASI EL DICCIONARIO FINAL:
key = pet image filename (ex: Beagle_01141.jpg)
value = List with:
index 0 = Pet Image Label (ex: beagle)
index 1 = Classifier Label (ex: english foxhound)
index 2 = 0/1 where 1 = labels match , 0 = labels don't match (ex: 0)
example_dictionary = {'Beagle_01141.jpg': ['beagle', 'walker hound, walker foxhound', 0]}"""

#AHORA METO TODO A UNA FUNCION
def adjust_results4_isadog(results_dic, dogfile):
    diccionario_perros_txt = dict()

    #ABRIMOS EL ARCHIVO TXT
    txt_dogs = open(dogfile, "r")
    #print (doc.read())
    #print (doc.readline())

    #READLINE() lee una a una las lineas del archivo, entonces lo recorremos hasta que este este vacio y lo metemos a una array primero
    #OJO READLINE SALTA UNA LINEA CADA VEZ QUE LO USO, SI LO LLAMO DOS VECES ME RECORRE DOS LINEAS
    txt_dogs_lista = []

    line = txt_dogs.readline()
    while line != "" :
        txt_dogs_lista.append( line.rstrip() )
        #rstrip quita los /n que salen por defecto cuando hay saltos de paginas
        line = txt_dogs.readline()
    txt_dogs.close()

    print("PRUEBA LISTA DEL DOGNAME - DEBERIA TENER 224 ELEMENTOS")
    print(len(txt_dogs_lista))

    #AHORA RECORREMOS TODO EL ARRAY ANTERIOR Y INGRESANDOLO AL NUEVO DICCIONARIO, SIEMPRE Y CUANDO NO ESTE

    for i in range(0, len(txt_dogs_lista), 1):
        if txt_dogs_lista[i] not in diccionario_perros_txt:
            #ACA ES DONDE LE DECIMOS, SI NO ESTA EN EL NUEVO DICCIONARIO, EN LA KEY CON NOMBRE DE LA LINEA DEL ARCHIVO INGRESA UN NUEVO CAMPO CON VALOR 1, ASI QUEDA
            #KEY = LINEA TXT; VALOR = 1
             diccionario_perros_txt[txt_dogs_lista[i]] = [1]

    #ENTONCES A ESTE PUNTO YA TENEMOS LISTO NUEVO NUESTRO DICCIONARIO DE PERROS DEL MUNDO ENTERO
    #AHORA EMPEZAMOS A HACER LA COMPARACION Y COMPROBAR SI NUESTRO CLASIFICADOR ACERTO O NO

    for key in results_dic:

        if results_dic[key][0] in diccionario_perros_txt:
            label_en_txt = 1
        else: label_en_txt = 0

        results_dic[key].append(label_en_txt)

        if results_dic[key][1] in diccionario_perros_txt:
            clasificador_en_txt = 1
        else: clasificador_en_txt = 0

        results_dic[key].append(clasificador_en_txt)

adjust_results4_isadog(results_dic, "dognames.txt")
print("SALIDA DEL ADJUST RESULT")
print(results_dic)


"""QUEDO ASI EL DICCIONARIO FINAL:
key = pet image filename (ex: Beagle_01141.jpg)
value = List with:
index 0 = Pet Image Label (ex: beagle)
index 1 = Classifier Label (ex: english foxhound)
index 2 = 0/1 where 1 = labels match , 0 = labels don't match (ex: 0)
index 3 = 0/1 where 1= Pet Image Label is a dog, 0 = Pet Image Label isn't a dog (ex: 1)
index 4 = 0/1 where 1= Classifier Label is a dog, 0 = Classifier Label isn't a dog (ex: 1)
example_dictionary = {'Beagle_01141.jpg': ['beagle', 'walker hound, walker foxhound', 0, 1, 1]}"""

#------------------------------------------------------------------------------------------------------------------------------------
#TO DO 6: ESTADISTICAS: HACER UN NUEVO DICCIONARIO CON LOS SIGUIENTES DATOS
"""
Number of Images
Number of Dog Images
Number of "Not-a" Dog Images
% Correctly Classified Dog Images
% Correctly Classified "Not-a" Dog Images
% Correctly Classified Breeds of Dog Images

SE VERIA ASI EL DICCIONARIO
key = statistic's name (e.g. n_correct_dogs, pct_correct_dogs, n_correct_breed, pct_correct_breed)
value = statistic's value (e.g. 30, 100%, 24, 80%)
example_dictionary = {'n_correct_dogs': 30, 'pct_correct_dogs': 100.0, 'n_correct_breed': 24, 'pct_correct_breed': 80.0}"""

results_stats_dic = dict()

    #INICIALIZAMOS LAS KEY Y VALOR QUE TENDRA EL DICCIONARIO DICCIONARIO[KEY]=VALOR
results_stats_dic['n_img_total'] = 0
results_stats_dic['n_pet_label_is_dog_img'] = 0
results_stats_dic['n_pet_label_is_not_dog_img'] = 0
nro_correct_clasif_dogs = 0
results_stats_dic['pct_correct_clasif_dogs'] = 0
nro_correct_clasif_not_dogs = 0
results_stats_dic['pct_correct_clasif_not_dogs'] = 0
nro_correct_breed = 0
results_stats_dic['pct_correct_breed'] = 0

results_stats_dic['n_img_total'] = len(results_dic)

for key in results_dic:

    if results_dic[key][3] == 1:
        results_stats_dic['n_pet_label_is_dog_img'] = results_stats_dic['n_pet_label_is_dog_img'] + 1

    if results_dic[key][3] == 0:
        results_stats_dic['n_pet_label_is_not_dog_img'] = results_stats_dic['n_pet_label_is_not_dog_img'] + 1

    if results_dic[key][3] == 1 and results_dic[key][4] == 1:
        nro_correct_clasif_dogs =  nro_correct_clasif_dogs + 1

    if results_dic[key][3] == 0 and results_dic[key][4] == 0:
        nro_correct_clasif_not_dogs  = nro_correct_clasif_not_dogs  + 1

    if results_dic[key][3] == 1 and results_dic[key][2] == 1:
        nro_correct_breed  = nro_correct_breed  + 1

    if results_stats_dic['n_pet_label_is_dog_img'] != 0:
        results_stats_dic['pct_correct_clasif_dogs'] = nro_correct_clasif_dogs / results_stats_dic['n_pet_label_is_dog_img'] * 100
    if results_stats_dic['n_pet_label_is_not_dog_img']  != 0:
        results_stats_dic['pct_correct_clasif_not_dogs'] = nro_correct_clasif_not_dogs / results_stats_dic['n_pet_label_is_not_dog_img'] * 100
    if results_stats_dic['n_pet_label_is_dog_img']  != 0:
        results_stats_dic['pct_correct_breed'] = nro_correct_breed / results_stats_dic['n_pet_label_is_dog_img'] * 100



print("ESTADISTICAS DICCIONARIO")
print(results_stats_dic)

#------------------------------------------------------------------------------------------------------------------------------
# TO DO 7: IMPRESION FINAL DE RESULTADOS

print("\n\n*** Results Summary for CNN Model Architecture",model.upper(),
      "***")
print("{:20}: {:3d}".format('N Images', results_stats_dic['n_img_total']))
print("{:20}: {:3d}".format('N Dog Images', results_stats_dic['n_pet_label_is_dog_img']))
print("{:20}: {:3d}".format('N not Dog Images', results_stats_dic['n_pet_label_is_not_dog_img']))

for key in results_stats_dic:
    if key[0] == "p":
        print(key , ":", results_stats_dic[key])

##IMPRIME LOS QUE SE CLASIFICARON MAL
if (results_stats_dic['pct_correct_clasif_dogs'] != 100 or results_stats_dic['pct_correct_clasif_not_dogs'] != 100 ):
    print("\nINCORRECT Dog/NOT Dog Assignments:")
    for key in results_dic:
        if results_dic[key][3] == 1 and results_dic[key][4] == 0:
            #print(key)
            print("Real: {:>26}   Classifier: {:>30}".format(results_dic[key][0], results_dic[key][1]))

        if results_dic[key][3] == 0 and results_dic[key][4] == 1:
            #print(key)
            print("Real: {:>26}   Classifier: {:>30}".format(results_dic[key][0], results_dic[key][1]))

if (results_stats_dic['pct_correct_breed']  != 100):
    print("\nINCORRECT Dog Breed Assignment:")
    for key in results_dic:
        if (sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0 ):
            print("Real: {:>26}   Classifier: {:>30}".format(results_dic[key][0], results_dic[key][1]))


#METEMOS TODO EN UNA FUNCION, OJO QUE LOS DOS ULTIMOS PARAMETROS SON PARA ACTIVAR SI QUEREMOS LOS DOS ULTIMOS PRINT, PARA ESO HABIA QUE COLOCARLE TRUE
def print_results(results_dic, results_stats_dic, model, print_incorrect_dogs = False, print_incorrect_breed = False):

    print("\n\n*** Results Summary for CNN Model Architecture",model.upper(),"***")
    print("{:20}: {:3d}".format('N Images', results_stats_dic['n_img_total']))
    print("{:20}: {:3d}".format('N Dog Images', results_stats_dic['n_pet_label_is_dog_img']))
    print("{:20}: {:3d}".format('N not Dog Images', results_stats_dic['n_pet_label_is_not_dog_img']))

    for key in results_stats_dic:
        if key[0] == "p":
            print(key , ":", results_stats_dic[key])

    ##IMPRIME LOS QUE SE CLASIFICARON MAL
    #RECORDAR QUE EN PYTHON SI COLOCO LA VARIABLE EN UN IF SIN NADA MAS ES COMO SI ESTUVIERA DICIENDO VARIABLE = TRUE
    if (print_incorrect_dogs and (results_stats_dic['pct_correct_clasif_dogs'] != 100 or results_stats_dic['pct_correct_clasif_not_dogs'] != 100 )):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        for key in results_dic:
            if results_dic[key][3] == 1 and results_dic[key][4] == 0:
                print(key)

            if results_dic[key][3] == 0 and results_dic[key][4] == 1:
                print(key)

    if (print_incorrect_breed and (results_stats_dic['pct_correct_breed']  != 100)):
        print("\nINCORRECT Dog Breed Assignment:")
        for key in results_dic:
            if (sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0 ):
                print("Real: {:>26}   Classifier: {:>30}".format(results_dic[key][0], results_dic[key][1]))
