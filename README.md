# Grepolympia Calculator

https://grepolympia-calculator.onrender.com

A web app that calculates the optimal point distribution for Summer Grepolympia events in Grepolis using machine learning.

## Features
- **Optimal Distribution:** Calculates the best way to allocate your athlete's points for maximum score in supported events.
- **Machine Learning:** Uses a Random Forest model trained on real event data to predict the best attribute allocation.
- **Modern UI:** Clean, responsive, and user-friendly interface.
- **Contact & Feedback:** Users can contact the developer via email or Discord for suggestions, questions, or to contribute more data.

## How it Works
- Select a Summer Grepolympia event and enter your available points.
- The app tries all possible ways to distribute the points among the attributes.
- It uses a trained Random Forest model to estimate the score for each combination.
- The best distribution and estimated score are shown to the user.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/TudorBuha/grepolympia-calculator.git
   cd grepolympia-calculator
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app/main.py
   ```
   Or use Flask:
   ```bash
   export FLASK_APP=app/main.py  # or set FLASK_APP=app/main.py on Windows
   flask run
   ```
4. Open your browser and go to `http://127.0.0.1:5000/`

## Documentation
See the [Documentation](https://github.com/TudorBuha/grepolympia-calculator#documentation) or click the "Documentation" button in the app for detailed usage instructions.

## Contributing & Feedback
- If you have suggestions, want to contribute data, or found a bug, please open an issue or contact me:
  - Email: grepolympia@gmail.com
  - Discord: tudor_buha
- If you have more event data that could improve the model, please reach out!

## License
MIT License

---

[GitHub Repository](https://github.com/TudorBuha/grepolympia-calculator)
