
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
from collections import defaultdict
import io
import json
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class VaccineScheduleOptimizer:
    def __init__(self):
        self.vaccine_effectiveness = defaultdict(lambda: defaultdict(list))
        self.vaccine_rankings = defaultdict(dict)
        self.historical_schedules = []
        
    def validate_data(self, df):
        required_columns = ['Age_wk', 'Impacting_Vaccine_ID', 'G_Mean', 'Virus_Test_Result', 'Mortality', 'Month']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        return True

    def preprocess_data(self, df):
        try:
            data = df.copy()
            
            # Convert strings to proper format
            data['Impacting_Vaccine_ID'] = data['Impacting_Vaccine_ID'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else 
                ([x] if x != 'Not In Any Vaccine Effect' else [])
            )
            
            # Convert numeric columns
            numeric_cols = ['Age_wk', 'Month', 'G_Mean', 'Mortality', 'Virus_Test_Result']
            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Fill missing values
            data['G_Mean'].fillna(data['G_Mean'].mean(), inplace=True)
            data['Mortality'].fillna(data['Mortality'].mean(), inplace=True)
            data['Virus_Test_Result'].fillna(0, inplace=True)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error preprocessing data: {str(e)}")

    def calculate_vaccine_impact(self, data, window_size=4):
        processed_data = self.preprocess_data(data)
        self.vaccine_effectiveness.clear()  # Clear previous data
        
        for idx, row in processed_data.iterrows():
            if idx < window_size:
                continue
                
            pre_window = processed_data.iloc[idx-window_size:idx]
            post_window = processed_data.iloc[idx:idx+window_size] if idx+window_size <= len(processed_data) else processed_data.iloc[idx:]
            
            if len(post_window) < 2:
                continue
            
            metrics = {
                'mortality_reduction': -(post_window['Mortality'].mean() - pre_window['Mortality'].mean()),
                'virus_resistance': 1 - post_window['Virus_Test_Result'].mean(),
                'g_mean_improvement': post_window['G_Mean'].mean() - pre_window['G_Mean'].mean() 
                if not pd.isna(post_window['G_Mean'].mean()) and not pd.isna(pre_window['G_Mean'].mean()) 
                else 0
            }
            
            composite_score = (
                0.4 * metrics['mortality_reduction'] +
                0.3 * metrics['virus_resistance'] +
                0.3 * metrics['g_mean_improvement']
            )
            
            age_week = row['Age_wk']
            month = row['Month']
            
            for vaccine in row['Impacting_Vaccine_ID']:
                if vaccine:  # Only add if vaccine is not empty
                    self.vaccine_effectiveness[age_week][vaccine].append({
                        'score': composite_score,
                        'metrics': metrics,
                        'month': month
                    })

    def rank_vaccines(self, min_observations=2):
        rankings = defaultdict(lambda: defaultdict(list))
        
        for age_week, vaccines in self.vaccine_effectiveness.items():
            vaccine_scores = []
            
            for vaccine, impacts in vaccines.items():
                if len(impacts) >= min_observations:
                    avg_score = np.mean([impact['score'] for impact in impacts])
                    month = impacts[0]['month']
                    vaccine_scores.append((vaccine, avg_score, month))
            
            if vaccine_scores:
                sorted_vaccines = sorted(vaccine_scores, key=lambda x: x[1], reverse=True)
                total_vaccines = len(sorted_vaccines)
                vaccines_per_tier = max(1, total_vaccines // 4)
                
                for rank in range(4):
                    start_idx = rank * vaccines_per_tier
                    end_idx = start_idx + vaccines_per_tier if rank < 3 else total_vaccines
                    
                    tier_vaccines = [v[0] for v in sorted_vaccines[start_idx:end_idx]]
                    if tier_vaccines:
                        rankings[age_week] = {
                            'month': sorted_vaccines[0][2],
                            f'rank_{rank+1}': tier_vaccines
                        }
        
        return rankings

    def optimize_schedule(self, data):
        """
        Generate optimized vaccine schedule with rankings
        """
        try:
            # Validate and preprocess data
            self.validate_data(data)
            processed_data = self.preprocess_data(data)
            
            # Calculate vaccine impact scores
            self.calculate_vaccine_impact(processed_data)
            
            # Get rankings
            rankings = self.rank_vaccines()
            
            # Convert rankings to schedule format
            schedule = []
            for age_week, rank_data in rankings.items():
                schedule_entry = {
                    'age_week': int(age_week),
                    'month': int(rank_data['month']),
                    'vaccines_by_rank': {
                        f'rank_{i+1}': rank_data.get(f'rank_{i+1}', [])
                        for i in range(4)
                    }
                }
                schedule.append(schedule_entry)
            
            # Sort by age week
            schedule = sorted(schedule, key=lambda x: x['age_week'])
            
            # Store in history
            self.historical_schedules.append({
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'schedule': schedule
            })
            
            return schedule
            
        except Exception as e:
            raise ValueError(f"Error generating schedule: {str(e)}")

    def reschedule(self, test_results, previous_schedule):
        """
        Modify schedule based on test results
        """
        try:
            updated_schedule = []
            
            for entry in previous_schedule:
                age_week = entry['age_week']
                test_result = next((r for r in test_results if r['age_week'] == age_week), None)
                
                if test_result and test_result.get('needs_adjustment'):
                    # Copy vaccines_by_rank to avoid modifying original
                    vaccines_by_rank = dict(entry['vaccines_by_rank'])
                    
                    # Get underperforming vaccines
                    underperforming = test_result.get('underperforming_vaccines', [])
                    
                    # Adjust rankings
                    for rank in range(1, 4):
                        current_rank = f'rank_{rank}'
                        next_rank = f'rank_{rank+1}'
                        
                        if current_rank in vaccines_by_rank and next_rank in vaccines_by_rank:
                            current_vaccines = list(vaccines_by_rank[current_rank])
                            next_vaccines = list(vaccines_by_rank[next_rank])
                            
                            for vaccine in underperforming:
                                if vaccine in current_vaccines:
                                    current_vaccines.remove(vaccine)
                                    next_vaccines.insert(0, vaccine)
                            
                            vaccines_by_rank[current_rank] = current_vaccines
                            vaccines_by_rank[next_rank] = next_vaccines
                    
                    updated_schedule.append({
                        'age_week': age_week,
                        'month': entry['month'],
                        'vaccines_by_rank': vaccines_by_rank
                    })
                else:
                    updated_schedule.append(entry)
            
            # Store in history
            self.historical_schedules.append({
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'schedule': updated_schedule
            })
            
            return updated_schedule
            
        except Exception as e:
            raise ValueError(f"Error rescheduling: {str(e)}")

# Initialize optimizer
optimizer = VaccineScheduleOptimizer()

# Rest of the Flask routes remain the same...
@app.route('/history')
def get_history():
    return jsonify({
        'status': 'success',
        'history': optimizer.historical_schedules
    })

if __name__ == '__main__':
    app.run(debug=True)