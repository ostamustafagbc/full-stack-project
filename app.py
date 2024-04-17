from flask import Flask, render_template

app = Flask(__name__, template_folder="resources/views")

@app.route('/')
def index():
    return render_template('main.html')