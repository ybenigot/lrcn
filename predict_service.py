from flask import Flask, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image
import simplecaptioner as sc

import os
os.environ["GLOG_minloglevel"] = "1"
import caffe

caffe.set_mode_cpu()

# load modellrcn-resnet_iter_95000
print 'load model...'
net = caffe.Net('deploy.prototxt', 1, weights='lrcn-resnet_iter_478261.caffemodel')
#net = caffe.Net('deploy.prototxt', 1, weights='../cvgj/resnet50/resnet50_cvgj_iter_215764.caffemodel')

print 'model loaded.'
print '--------------------------------------------------------------------------------'

VOCAB_FILE = 'vocabulary.txt'

captioner=sc.SimpleCaptioner(net,VOCAB_FILE)

################################ Flask server #######################
# web server parameters
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = set(['JPG', 'JPEG', 'jpg', 'jpeg'])

# presentation
header = '<!doctype html><html><body>'
html='<form action="/predict" method="post" enctype="multipart/form-data"><input type=file name="file"><br/><input type="submit"></form>'
footer='</body></html>'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize(filename):
	im = Image.open(filename)
	im2 = im.resize((256,256))
	im2.save(filename)

# call caffe to make a prediction on the uploaded file
@app.route('/predict',methods=['GET','POST'])
def predict():
	if request.method == 'POST':
		# save file to disk
		if 'file' not in request.files:
			print('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			print('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(file_path)	
		else:
			print('File not allowed')
			return redirect(request.url)		
		# read image and resize it
		resize(file_path)
		im = np.array(Image.open(file_path))
		#crop image
		im2 = caffe.io.oversample((im,),(224,224))
		#take first crop
		net.blobs['data'].data[...] = np.transpose(im2[0:1,:,:,:], (0,3,1,2))
		# forward propagation for images
		net.forward(start='data',end='score')
		image_rep  = net.blobs['score'].data[0,:]

		# caption sentence generation
		sentence = captioner.sample_caption(image_rep)
		print sentence

		response=header
		response += '<img src="/image/' + filename + '"></img><br>'
		response += '<br/><p>'+ captioner.get_sentence(sentence) +'</p><br/>'

		return response + '<br>' + html + footer
	else:
		return header + html + footer

# display the uploaded image
@app.route('/image/<filename>',methods=['GET'])
def image(filename):
	return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')

