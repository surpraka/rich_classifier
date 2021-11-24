# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:53:17 2021

@author: suraj
"""

from flask import Flask, request, jsonify,render_template
import server.util as util

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    image_data = request.form['image_data']
    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/')
def home():
    return render_template('app.html')

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(debug=True)