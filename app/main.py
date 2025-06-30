import os
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from app.model import load_dataset, train_and_select_model, predict_best_distribution
from datetime import datetime
from flask_mail import Mail, Message

# app = Flask(__name__)
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates'), static_folder=os.path.join(os.path.dirname(__file__), '../static'))
app.config["UPLOAD_FOLDER"] = "app/data"

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'grepolympia@gmail.com'  # update if needed
app.config['MAIL_PASSWORD'] = 'YOUR_APP_PASSWORD_HERE'  # to be filled in by user
app.config['MAIL_DEFAULT_SENDER'] = 'grepolympia@gmail.com'

mail = Mail(app)

# Mapă probe (va fi folosită și în frontend)
discipline_files = {
    "Hoplite Race": "hoplite_race.xlsx",
    "Archery": "archery.xlsx",
    "Javelin Throw": "javelin_throw.xlsx"
}

# Attribute names per event
event_attributes = {
    "Hoplite Race": ["Speed", "Strength", "Endurance"],
    "Archery": ["Concentration", "Intuition", "Accuracy"],
    "Javelin Throw": ["Momentum", "Technique", "Throwing Power"]
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", disciplines=discipline_files.keys(), year=datetime.now().year)

@app.route("/predict", methods=["POST"])
def predict():
    discipline = request.form["discipline"]
    level = int(request.form["level"])

    if level > 500:
        error = "Input too large: The model runs very slow on big values. Please enter a value of 500 or less."
        return render_template("index.html",
                               disciplines=discipline_files.keys(),
                               selected=discipline,
                               level=level,
                               error=error,
                               year=datetime.now().year)

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], discipline_files[discipline])
    df = load_dataset(file_path)
    model_tuple = train_and_select_model(df)
    dist, score, extrapolation = predict_best_distribution(model_tuple, level, df)

    # Map result keys to attribute names for the selected event
    attr_names = event_attributes[discipline]
    result_named = {attr_names[i]: v for i, v in enumerate(dist.values())}

    return render_template("index.html",
                           disciplines=discipline_files.keys(),
                           selected=discipline,
                           level=level,
                           result=result_named,
                           score=score,
                           extrapolation=extrapolation,
                           year=datetime.now().year)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    new_name = request.form["new_name"]

    if file and new_name:
        filename = secure_filename(new_name.replace(" ", "_").lower() + ".xlsx")
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        # Adaugă în mapă
        discipline_files[new_name] = filename

    return redirect("/")

@app.route("/contact", methods=["POST"])
def contact():
    return render_template("index.html", disciplines=discipline_files.keys(), contact_success=True, year=datetime.now().year)

@app.route("/docs")
def docs():
    return render_template("documentation.html")
