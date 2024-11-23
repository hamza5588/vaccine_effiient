from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

app = Flask(__name__)

# Global data and models
data = pd.DataFrame()
regressor_gmean = RandomForestRegressor(n_estimators=100, random_state=42)
regressor_mortality = RandomForestRegressor(n_estimators=100, random_state=42)
classifier_virus_test = RandomForestClassifier(n_estimators=100, random_state=42)
le = LabelEncoder()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global data, regressor_gmean, regressor_mortality, classifier_virus_test, le
    if request.method == 'POST':
        file = request.files['file']
        if file:
            data = pd.read_csv(file)
            data = data.sample(frac=0.5, random_state=42)  # Use 50% of the data for faster initial testing

            data = data.dropna(subset=['G_Mean', 'Mortality', 'Virus_Test_Result'])
            data['Vaccine_ID_Encoded'] = le.fit_transform(data['Impacting_Vaccine_ID'].astype(str))
            X = data[['Age_wk', 'Month', 'Vaccine_ID_Encoded']]
            y_gmean = data['G_Mean']
            y_mortality = data['Mortality']
            y_virus_test = data['Virus_Test_Result']

            # Train models
            regressor_gmean.fit(X, y_gmean)
            regressor_mortality.fit(X, y_mortality)
            classifier_virus_test.fit(X, y_virus_test)

            return redirect(url_for('display_all'))
    return render_template('upload.html')

@app.route('/display_all')
def display_all():
    if data.empty:
        return "No data available. Please upload a dataset."
    
    all_ranked_schedules = []
    for age_week in range(1, 95):  # Adjust based on your data range (e.g., 1 to 94 weeks)
        for month in data['Month'].unique():  # Iterate through unique months
            ranked_data = predict_and_rank_vaccines(age_week, month)
            ranked_data['Age_wk'] = age_week
            ranked_data['Month'] = month
            all_ranked_schedules.append(ranked_data)

    all_schedules_df = pd.concat(all_ranked_schedules)
    all_schedules_df = all_schedules_df.sort_values(by=['Age_wk', 'Month', 'Rank'])

    # Save the result as a CSV file
    result_file_path = 'static/ranked_vaccine_schedule.csv'
    all_schedules_df.to_csv(result_file_path, index=False)

    return render_template('display_all.html', tables=[all_schedules_df.to_html()], titles=['Full Vaccine Schedule'], file_path=result_file_path)

@app.route('/download')
def download_file():
    file_path = 'static/ranked_vaccine_schedule.csv'
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found."

@app.route('/input_new_data', methods=['GET', 'POST'])
def input_new_data():
    global data
    if request.method == 'POST':
        age_week = int(request.form['age_week'])
        month = int(request.form['month'])
        mortality = float(request.form['mortality'])
        g_mean = float(request.form['g_mean'])
        virus_test_result = int(request.form['virus_test_result'])
        
        new_row = {
            'Age_wk': age_week,
            'Month': month,
            'Mortality': mortality,
            'G_Mean': g_mean,
            'Virus_Test_Result': virus_test_result,
            'Impacting_Vaccine_ID': 'New Data'
        }
        data = data.append(new_row, ignore_index=True)
        return redirect(url_for('upload'))
    return render_template('input_new_data.html')

def predict_and_rank_vaccines(age_week, month):
    vaccine_ids_encoded = np.unique(data['Vaccine_ID_Encoded'])
    predictions = []

    for vaccine_id in vaccine_ids_encoded:
        input_data = np.array([[age_week, month, vaccine_id]])
        predicted_gmean = regressor_gmean.predict(input_data)[0]
        predicted_mortality = regressor_mortality.predict(input_data)[0]
        predicted_virus_test = classifier_virus_test.predict_proba(input_data)[0][1]

        # Include only if the test result is negative (0)
        if predicted_virus_test == 0:
            vaccine_id_decoded = le.inverse_transform([vaccine_id])[0]

            predictions.append({
                'Vaccine_ID': vaccine_id_decoded,
                'Predicted_G_Mean': predicted_gmean,
                'Predicted_Mortality': predicted_mortality
            })

    # Return an empty DataFrame if no valid predictions are found
    if not predictions:
        return pd.DataFrame(columns=['Vaccine_ID', 'Predicted_G_Mean', 'Predicted_Mortality', 'Age_wk', 'Month', 'Rank'])

    # Create a DataFrame and rank vaccines by G_Mean (higher is better)
    predictions_df = pd.DataFrame(predictions)
    predictions_df['Rank'] = predictions_df['Predicted_G_Mean'].rank(ascending=False, method='dense')
    return predictions_df.sort_values(by='Rank')

if __name__ == '__main__':
    app.run(debug=True)
