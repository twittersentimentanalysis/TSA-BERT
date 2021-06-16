import json
import Classifier
import Initialization

from functools      import wraps
from flask_restful  import Resource, Api
from flask          import Flask, request, jsonify, abort

app = Flask(__name__)
api = Api(app)

# The actual decorator function
def require_appkey(view_function):
	@wraps(view_function)
	# the new, post-decoration function. Note *args and **kwargs here.
	def decorated_function(*args, **kwargs):
		with open('api.key', 'r') as apikey:
			key = apikey.read().replace('\n', '')
		if request.headers.get('x-api-key') and request.headers.get('x-api-key') == key:
			return view_function(*args, **kwargs)
		else:
			abort(401)
	return decorated_function

class Emotion(Resource):
	@require_appkey
	def post(self):
		text = request.json['text']
		bert = request.json['bert']
		model, config, label_dict = Initialization.load_model(bert)
		emotions = Classifier.get_emotion(text, model, bert, config, label_dict)
		return emotions

# Routes
api.add_resource(Emotion, '/bert/v1/emotion')  

# Main
if __name__ == '__main__':
	app.run(host='0.0.0.0', port='6231')