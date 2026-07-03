from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'smartbridge_secret_hdi_key'  # Required to securely track User Sessions

# 1. Mock Database to handle User and Session ER entities
users_db = {
    "admin@hdi.com": {"name": "Kavyamrutha", "password": "password123", "role": "Evaluator"}
}

# 2. Load the trained machine learning model
try:
    model = pickle.load(open('HDI.pkl', 'rb'))
except FileNotFoundError:
    model = None
    print("Warning: HDI.pkl file not found! Please run 'python main.py' first.")

def classify_hdi(score):
    """Maps the predicted continuous regression score to the 4 mandated tiers."""
    score = max(0.0, min(1.0, score))
    if score >= 0.800: return f"{score:.3f} - Very High Human Development"
    elif score >= 0.700: return f"{score:.3f} - High Human Development"
    elif score >= 0.550: return f"{score:.3f} - Medium Human Development"
    else: return f"{score:.3f} - Low Human Development"

# ---------------------------------------------------------------------
# ROUTING & CONTROLLERS
# ---------------------------------------------------------------------

@app.route('/')
def index():
    # If user is already logged in, send them straight to the Home dashboard
    if 'user_email' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if email in users_db and users_db[email]['password'] == password:
            session['user_email'] = email
            session['user_name'] = users_db[email]['name']
            return redirect(url_for('home'))
        else:
            error = "Invalid email or password parameters. Please try again."
            
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    message = None
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        if email in users_db:
            message = "Account registration failed: Email already exists."
        else:
            users_db[email] = {"name": name, "password": password, "role": "User"}
            message = "Account created successfully! You can now log in."
            
    return render_template('register.html', message=message)

@app.route('/home')
def home():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', user_name=session['user_name'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_email' not in session:
        return redirect(url_for('login'))
        
    try:
        features = [
            float(request.form['Life_Expectancy']),
            float(request.form['Expected_Years_Schooling']),
            float(request.form['GNI_Per_Capita']),
            float(request.form['Internet_Users_Pct']),
            float(request.form['Carbon_Emission_Per_Capita'])
        ]
        
        if model is None:
            return render_template('home.html', user_name=session['user_name'], prediction_text="Model file 'HDI.pkl' is missing.")
            
        raw_prediction = model.predict([np.array(features)])[0]
        result_tier = classify_hdi(raw_prediction)
        
        return render_template('home.html', user_name=session['user_name'], prediction_text=f'Predicted Tier: {result_tier}')
        
    except Exception as e:
        return render_template('home.html', user_name=session['user_name'], prediction_text=f'Error Processing Input: {str(e)}')

@app.route('/logout')
def logout():
    # Destroys the user session block entirely
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)