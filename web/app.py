import os
import cv2
import jsonpickle
import numpy as np
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from tensorflow.keras.models import model_from_json

app = Flask(__name__, template_folder='templates')
# UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')


with open('modelos/melhor_modelo.json', 'r') as json_file:
    model = json_file.read()

model = model_from_json(model)
model.load_weights('modelos/melhor_peso.best.hdf5')


def upload_image():
    filestr = request.files['imagem'].read()
    np_image = np.fromstring(filestr, np.uint8)

    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255

    tam = 128
    image = cv2.resize(image, (tam, tam))

    test_image = np.array([image])

    result = []
    prediction = model.predict_on_batch(test_image)
    result.append(prediction)

    result = np.asarray(result)
    imprime = np.array(result[0][0])

    return np.argmax(imprime)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    image = upload_image()

    monkeys = ['mantled howler',
               'patas monkey',
               'bald uakari',
               'japanese macaque',
               'pygmy marmoset',
               'white headed capuchin',
               'silvery marmoset',
               'common squirrel monkey',
               'black headed night monkey',
               'nilgiri langur']

    return render_template('index.html', text=str(monkeys[image].upper()))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
