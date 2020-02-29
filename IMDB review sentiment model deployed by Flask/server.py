from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import tensorflow as tf
from model_NLP import comment_preprocess


app = Flask(__name__)
api = Api(app)

model = tf.keras.models.load_model('./model.h5')

class PredictComment(Resource):
    def get(self):
        return {'about':'Hi!Please use POST method to predict your comment!'}
    def post(self):
        data = request.get_json(force=True)
        comment = data['comment']
        train_data = comment_preprocess(comment)
        result = model.predict([train_data])
        return {'positive_rate': result.tolist()[0][0]}

api.add_resource(PredictComment,'/predict')

if __name__ == '__main__':
    app.run(port=5000, debug=True)


###basic rest api method
# @app.route('/api', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     comment = data['comment']
#     train_data = comment_preprocess(comment)
#     result = model.predict([train_data])
#     print(type(result))
#     return {'positive_rate':result.tolist()[0][0]}


