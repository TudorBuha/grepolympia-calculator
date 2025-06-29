import os
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from app.model import load_dataset, train_model, predict_best_distribution

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "app/data"

# Mapă probe (va fi folosită și în frontend)
discipline_files = {
    "Hoplite Race": "hoplite_race.xlsx",
    "Archery": "archery.xlsx"
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", disciplines=discipline_files.keys())

@app.route("/predict", methods=["POST"])
def predict():
    discipline = request.form["discipline"]
    level = int(request.form["level"])

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], discipline_files[discipline])
    df = load_dataset(file_path)
    model = train_model(df)
    dist, score = predict_best_distribution(model, level)

    return render_template("index.html",
                           disciplines=discipline_files.keys(),
                           selected=discipline,
                           level=level,
                           result=dist,
                           score=score)

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
