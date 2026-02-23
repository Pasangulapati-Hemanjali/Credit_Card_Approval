from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "super_secret_key_123"

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ================= LOAD MODEL FILES =================
with open(os.path.join(MODEL_DIR, "credit_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "features.pkl"), "rb") as f:
    feature_names = pickle.load(f)

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")


# ---------- SIGNUP ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        session["username"] = request.form["username"]
        return redirect(url_for("dashboard"))
    return render_template("signup.html")


# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin@gmail.com" and password == "admin123":
            session["username"] = username
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")


# ---------- DASHBOARD ----------
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])


# ---------- APPLY ----------
@app.route("/apply")
def apply():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("apply.html")


# ---------- CONFIRM PAGE ----------
@app.route("/page2", methods=["POST"])
def page2():
    if "username" not in session:
        return redirect(url_for("login"))

    form_data = request.form.to_dict()
    return render_template("page2.html", data=form_data)


# ---------- PREDICT ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "username" not in session:
        return redirect(url_for("login"))

    data = request.form

    input_data = {
        "AMT_INCOME_TOTAL": float(data["income"]),
        "CNT_CHILDREN": int(data["children"]),
        "CNT_FAM_MEMBERS": int(data["family"]),
        "DAYS_EMPLOYED": int(data["employment_days"]),
        "CREDIT_SCORE": int(data["credit_score"]),
        "CODE_GENDER_M": 1 if data["gender"] == "M" else 0,
        "FLAG_OWN_CAR_Y": 1 if data.get("own_car") == "Y" else 0,
        "FLAG_OWN_REALTY_Y": 1 if data.get("own_house") == "Y" else 0,
    }

    categorical_fields = {
        "NAME_INCOME_TYPE": data.get("income_type"),
        "NAME_EDUCATION_TYPE": data.get("education"),
        "OCCUPATION_TYPE": data.get("occupation"),
        "NAME_HOUSING_TYPE": data.get("housing"),
    }

    for col, val in categorical_fields.items():
        if val:
            col_name = f"{col}_{val}"
            if col_name in feature_names:
                input_data[col_name] = 1

    df = pd.DataFrame([input_data])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = int(model.predict_proba(df_scaled)[0][1] * 100)

    result = "APPLICATION APPROVED ✅" if prediction == 0 else "APPLICATION REJECTED ❌"
    risk = "LOW RISK" if prediction == 0 else "HIGH RISK"

    return render_template("result.html", result=result, risk=risk, probability=probability)


# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)

