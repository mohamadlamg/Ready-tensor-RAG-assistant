from flask import Flask,request,jsonify
from toch_utils import transform_images,get_predict

app = Flask(__name__)


allowed_extensions = {'jpg','png','jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in allowed_extensions 


@app.route('/predict',methods=['POST'])
def predict() :

    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error':'No file'})
        if not allowed_file(file.filename):
            return jsonify({'error':'Format not supported'})
        
        try:
            images_bytes = file.read()
            tensor = transform_images(images_bytes)
            prediction = get_predict(tensor)
            data = {'Prediction':prediction.item(),'class_name':str(prediction.item())}
            return jsonify(data)



        except:
            return jsonify({'error':'Erreur pendant la prediction'})




if __name__ == "__main__":
    print("Démarrage du serveur Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)