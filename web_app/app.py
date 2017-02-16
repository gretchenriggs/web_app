from flask import Flask, render_template
# from flask.ext.sqlalchemy import SQLAlchemy
import os
import cPickle as pickle

app = Flask(__name__)
# app.config.from_object(os.environ['APP_SETTINGS'])
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False




# from models import Result

# home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return 'dashboard'
    # Read data from Postgres

@app.route('/score', methods = ['GET', 'POST'])
def score():
    # Will want to call prediction script here
    return render_template('score.html')

if __name__ == '__main__':
    with open("../data/random_forest.pkl") as f_un:
        model = pickle.load(f_un)
    # db = SQLAlchemy(app)
    app.run(host='0.0.0.0', port=8105, debug=True)
