from django.conf import settings
from django.shortcuts import render
from .models import InsuranceClaim
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import sklearn
import warnings
warnings.filterwarnings('ignore')





# Load the saved model and scalers
model = joblib.load('voting_classifier_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')







def kpi_metrics(request):
    # Load the model (if needed for predictions)
    model = joblib.load('voting_classifier_model.pkl')

    # Get all claims
    claims = InsuranceClaim.objects.all()
    data = pd.DataFrame(list(claims.values()))

    # Convert incident_date to datetime
    data['incident_date'] = pd.to_datetime(data['incident_date'])

    # Get min and max dates
    min_date = data['incident_date'].min().strftime('%Y-%m-%d')
    max_date = data['incident_date'].max().strftime('%Y-%m-%d')

    # Filter data based on selected time period
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    if start_date and end_date:
        # Filter the data based on the selected dates
        data = data[(data['incident_date'] >= start_date) & (data['incident_date'] <= end_date)]

    # Initialize a dictionary to store KPIs
    kpis = {}

    # Customer Metrics
    kpis['Total Customers'] = len(data)
    kpis['Fraud Cases'] = data['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0).sum()
    kpis['Legit Cases'] = kpis['Total Customers'] - kpis['Fraud Cases']
    kpis['Average Age of Policyholders'] = round(data['age'].mean(), 2)
    kpis['Average Premium'] = round(data['policy_annual_premium'].mean(), 2)
    kpis['Average Duration as Customer'] = round(data['months_as_customer'].mean(), 2)
    kpis['Max Duration as Customer'] = round(data['months_as_customer'].max(), 2)
    kpis['Min Duration as Customer'] = round(data['months_as_customer'].min(), 2)
    kpis['Fraud Rate (%)'] = round((kpis['Fraud Cases'] / kpis['Total Customers']) * 100, 2)
    
    # Policy Metrics
    kpis['Average Policy Deductible'] = round(data['policy_deductable'].mean(), 2)
    kpis['Policies by State'] = data['policy_state'].value_counts().to_dict()
    kpis['Policy CSL Distribution'] = data['policy_csl'].value_counts().to_dict()
    kpis['Average Umbrella Limit'] = round(data['umbrella_limit'].mean(), 2)
    kpis['Claims by Policy Deductible'] = data.groupby('policy_deductable')['total_claim_amount'].sum().to_dict()
    
    # Claim Metrics
    kpis['Total Claims Amount'] = round(data['total_claim_amount'].sum(), 2)
    kpis['Claim Frequency'] = round(kpis['Total Claims Amount'] / kpis['Total Customers'], 2)
    kpis['Claim Severity'] = round(kpis['Total Claims Amount'] / (data[['injury_claim', 'property_claim', 'vehicle_claim']].sum().sum()), 2)
    kpis['Average Injury Claim'] = round(data['injury_claim'].mean(), 2)
    kpis['Average Property Claim'] = round(data['property_claim'].mean(), 2)
    kpis['Average Vehicle Claim'] = round(data['vehicle_claim'].mean(), 2)
    
    # Incident Metrics
    kpis['Incident Frequency'] = len(data)
    # Calculate mean values and round them
    incident_severity_means = data.groupby('incident_severity')['total_claim_amount'].mean()
    rounded_means = incident_severity_means.round(2).to_dict()

    # Update kpis dictionary
    kpis['Incident Severity'] = rounded_means

    kpis['Claims by Incident Type'] = data['incident_type'].value_counts().to_dict()
    kpis['Property Damage Rate (%)'] = round(data['property_damage'].value_counts(normalize=True).get('YES', 0) * 100, 2)
    kpis['Average Vehicles Involved'] = round(data['number_of_vehicles_involved'].mean(), 2)
    kpis['Average Incident Hour'] = round(data['incident_hour_of_the_day'].mean(), 2)
    kpis['Incident State Distribution'] = data['incident_state'].value_counts().to_dict()
    
    # Vehicle Metrics
    kpis['Most Common Vehicle Makes'] = data['auto_make'].value_counts().to_dict()
    kpis['Most Common Vehicle Models'] = data['auto_model'].value_counts().to_dict()
    kpis['Average Vehicle Age'] = round(2024 - data['auto_year'].mean(), 2)
    kpis['Vehicle Make Distribution'] = data['auto_make'].value_counts().to_dict()
    
    
    # Demographic Metrics
    kpis['Distribution by Sex'] = data['insured_sex'].value_counts().to_dict()
    kpis['Education Level Analysis'] = data['insured_education_level'].value_counts().to_dict()
    kpis['Occupation Analysis'] = data['insured_occupation'].value_counts().to_dict()
    kpis['Hobbies Analysis'] = data['insured_hobbies'].value_counts().to_dict()
    kpis['Relationship Status Distribution'] = data['insured_relationship'].value_counts().to_dict()
    
    # Incident Location Metrics
    kpis['Incident Location Analysis'] = data.groupby(['incident_state', 'incident_city']).size().to_dict()
    kpis['Top Incident Cities'] = data['incident_city'].value_counts().head(10).to_dict()
    kpis['Top Incident States'] = data['incident_state'].value_counts().head(10).to_dict()
    
    # Operational Metrics
    kpis['Average Number of Vehicles Involved'] = round(data['number_of_vehicles_involved'].mean(), 2)
    kpis['Police Report Availability Rate (%)'] = round(data['police_report_available'].value_counts(normalize=True).get('YES', 0) * 100, 2)
    kpis['Total Witnesses'] = data['witnesses'].sum()
    kpis['Total Capital Gains'] = round(data['capital_gains'].sum(), 2)
    kpis['Total Capital Loss'] = round(data['capital_loss'].sum(), 2)
    
    # Customer Behavior Metrics
    kpis['Average Capital Gains'] = round(data['capital_gains'].mean(), 2)
    kpis['Average Capital Loss'] = round(data['capital_loss'].mean(), 2)
    kpis['Average Policy Binding Month'] = round(data['policy_bind_date'].dt.month.mean(), 2)
    kpis['Average Capital Gains per Fraud Case'] = round(data[data['fraud_reported'] == 'Y']['capital_gains'].mean(), 2)
    kpis['Average Capital Loss per Fraud Case'] = round(data[data['fraud_reported'] == 'Y']['capital_loss'].mean(), 2)

    # Chart Data for Chart.js
    chart_data_1 = {
        'customer_age_distribution': {
            'labels': data['age'].unique().tolist(), 
            'values': data['age'].value_counts().tolist() 
        },
        'premium_vs_age': {
            'labels': data['age'].tolist(), 
            'values': data['policy_annual_premium'].tolist() 
        },
        'fraud_rate_over_time': {
            'labels': pd.to_datetime(data['incident_date']).dt.to_period('M').astype(str).unique().tolist(),
            'values': (data.groupby(pd.to_datetime(data['incident_date']).dt.to_period('M'))['fraud_reported']
                        .apply(lambda x: (x == 'Y').mean() * 100)).tolist()
        },
        'duration_vs_premium': {
            'labels': data['months_as_customer'].tolist(), 
            'values': data['policy_annual_premium'].tolist() 
        },
        'policies_by_state': {
            'labels': list(kpis['Policies by State'].keys()),
            'values': list(kpis['Policies by State'].values())
        },
        'policy_csl_distribution': {
            'labels': list(kpis['Policy CSL Distribution'].keys()),
            'values': list(kpis['Policy CSL Distribution'].values())
        },
        'deductible_vs_premium': {
            'labels': data['policy_deductable'].tolist(), 
            'values': data['policy_annual_premium'].tolist() 
        },
        'umbrella_limit_distribution': {
            'labels': data['umbrella_limit'].unique().tolist(),
            'values': data['umbrella_limit'].value_counts().tolist()
        },
        'total_claims_over_time': {
            'labels': pd.to_datetime(data['incident_date']).dt.to_period('M').astype(str).unique().tolist(),
            'values': data.groupby(pd.to_datetime(data['incident_date']).dt.to_period('M'))['total_claim_amount'].sum().tolist()
        },
        'claim_severity_vs_type': {
            'labels': data['incident_type'].unique().tolist(),
            'values': data.groupby('incident_type')['total_claim_amount'].mean().tolist()
        },
        'claim_amount_distribution': {
            'labels': data['total_claim_amount'].unique().tolist(),
            'values': data['total_claim_amount'].value_counts().tolist()
        },
        'claims_by_incident_type': {
            'labels': list(kpis['Claims by Incident Type'].keys()),
            'values': list(kpis['Claims by Incident Type'].values())
        },
        'claim_amount_by_deductible': {
            'labels': data['policy_deductable'].unique().tolist(),
            'values': data.groupby('policy_deductable')['total_claim_amount'].mean().tolist()
        },

        'property_damage_rate': {
            'labels': ['YES', 'NO'],
            'values': data['property_damage'].value_counts().tolist()
        },
        # 'incident_severity_vs_type': {
        #     'labels': data['incident_type'].unique().tolist(),
        #     'values': data.groupby('incident_type')['incident_severity'].mean().tolist()
        # },
        'average_vehicles_involved_over_time': {
            'labels': pd.to_datetime(data['incident_date']).dt.to_period('M').astype(str).unique().tolist(),
            'values': data.groupby(pd.to_datetime(data['incident_date']).dt.to_period('M'))['number_of_vehicles_involved'].mean().tolist()
        },
        'incident_frequency_over_time': {
            'labels': pd.to_datetime(data['incident_date']).dt.to_period('M').astype(str).unique().tolist(),
            'values': data.groupby(pd.to_datetime(data['incident_date']).dt.to_period('M')).size().tolist()
        },
        'most_common_vehicle_makes': {
            'labels': list(kpis['Most Common Vehicle Makes'].keys()),
            'values': list(kpis['Most Common Vehicle Makes'].values())
        },
        'vehicle_age_distribution': {
            'labels': data['auto_year'].unique().tolist(), 
            'values': data['auto_year'].value_counts().tolist() 
        },
        'claim_amount_vs_vehicle_age': {
            'labels': data['auto_year'].tolist(), 
            'values': data['total_claim_amount'].tolist() 
        },
        'demographic_distribution': {
            'labels': ['Male', 'Female'],
            'values': data['insured_sex'].value_counts().tolist()
        },
        'claims_by_education_level': {
            'labels': list(kpis['Education Level Analysis'].keys()),
            'values': list(kpis['Education Level Analysis'].values())
        },
        'claims_by_occupation': {
            'labels': list(kpis['Occupation Analysis'].keys()),
            'values': list(kpis['Occupation Analysis'].values())
        },
        'incident_location_analysis': {
            'labels': list(kpis['Incident Location Analysis'].keys()),
            'values': list(kpis['Incident Location Analysis'].values())
        },
        'top_incident_cities': {
            'labels': list(kpis['Top Incident Cities'].keys()),
            'values': list(kpis['Top Incident Cities'].values())
        },
        'top_incident_states': {
            'labels': list(kpis['Top Incident States'].keys()),
            'values': list(kpis['Top Incident States'].values())
        },
        'average_vehicles_involved_over_time': {
            'labels': pd.to_datetime(data['incident_date']).dt.to_period('M').astype(str).unique().tolist(),
            'values': data.groupby(pd.to_datetime(data['incident_date']).dt.to_period('M'))['number_of_vehicles_involved'].mean().tolist()
        },
        'police_report_availability': {
            'labels': ['YES', 'NO'],
            'values': data['police_report_available'].value_counts().tolist()
        },
        'claim_amount_vs_witnesses': {
            'labels': data['witnesses'].tolist(),
            'values': data['total_claim_amount'].tolist()
        },
        'average_capital_gains_over_time': {
            'labels': pd.to_datetime(data['incident_date']).dt.to_period('M').astype(str).unique().tolist(),
            'values': data.groupby(pd.to_datetime(data['incident_date']).dt.to_period('M'))['capital_gains'].mean().tolist()
        },
        'capital_gains_vs_claim_amount': {
            'labels': data['capital_gains'].tolist(),
            'values': data['total_claim_amount'].tolist()
        },
        'policy_binding_month_over_time': {
            'labels': pd.to_datetime(data['policy_bind_date']).dt.to_period('M').astype(str).unique().tolist(),
            'values': data.groupby(pd.to_datetime(data['policy_bind_date']).dt.to_period('M')).size().tolist()
        }
    }

    # Format KPI values
    kpis_formatted = {
        'Total_Customers': f"{kpis['Total Customers']}",
        'Fraud_Cases': f"{kpis['Fraud Cases']}",
        'Legit_Cases': f"{kpis['Legit Cases']}",
        'Average_Age_of_Policyholders': f"{kpis['Average Age of Policyholders']} years",
        'Average_Premium': f"${kpis['Average Premium']:,.2f}",
        'Average_Duration_as_Customer': f"{kpis['Average Duration as Customer']} months",
        'Max_Duration_as_Customer': f"{kpis['Max Duration as Customer']} months",
        'Min_Duration_as_Customer': f"{kpis['Min Duration as Customer']} months",
        'Fraud_Rate': f"{kpis['Fraud Rate (%)']}%",
        'Average_Policy_Deductible': f"${kpis['Average Policy Deductible']:,.2f}",
        'Policies_by_State': kpis['Policies by State'],
        'Average_Umbrella_Limit': f"${kpis['Average Umbrella Limit']:,.2f}",
        'Claims_by_Policy_Deductible': kpis['Claims by Policy Deductible'],
        'Total_Claims_Amount': f"${kpis['Total Claims Amount']:,.2f}",
        'Claim_Frequency': f"${kpis['Claim Frequency']:,.2f}",
        'Claim_Severity': f"${kpis['Claim Severity']:,.2f}",
        'Incident_Frequency': f"{kpis['Incident Frequency']}",
        'Property_Damage_Rate': f"{kpis['Property Damage Rate (%)']}%",
        'Average_Vehicle_Age': f"{kpis['Average Vehicle Age']} years",
        'Average_Number_of_Vehicles_Involved': f"{kpis['Average Number of Vehicles Involved']}",
        'Police_Report_Availability_Rate': f"{kpis['Police Report Availability Rate (%)']}%",
        'Total_Witnesses': f"{kpis['Total Witnesses']}",
        'Average_Capital_Gains': f"${kpis['Average Capital Gains']:,.2f}",
        'Average_Capital_Loss': f"${kpis['Average Capital Loss']:,.2f}",
        'Average_Policy_Binding_Month': f"{kpis['Average Policy Binding Month']}",
        'Average_Capital_Gains_per_Fraud_Case': f"${kpis['Average Capital Gains per Fraud Case']:,.2f}",
        'Average_Capital_Loss_per_Fraud_Case': f"${kpis['Average Capital Loss per Fraud Case']:,.2f}",
        'Most_Common_Vehicle_Makes': kpis['Most Common Vehicle Makes'],
        'Most_Common_Vehicle_Models': kpis['Most Common Vehicle Models'],
        'Vehicle_Make_Distribution': kpis['Vehicle Make Distribution'],
        'Distribution_by_Sex': kpis['Distribution by Sex'],
        'Education_Level_Analysis': kpis['Education Level Analysis'],
        'Occupation_Analysis': kpis['Occupation Analysis'],
        'Average_Injury_Claim': f"${kpis['Average Injury Claim']:,.2f}",
        'Average_Property_Claim': f"${kpis['Average Property Claim']:,.2f}",
        'Average_Vehicle_Claim': f"${kpis['Average Vehicle Claim']:,.2f}",
        'Incident_Severity': kpis['Incident Severity'],
        'Claims_by_Incident_Type': kpis['Claims by Incident Type'],
        'Incident_State_Distribution': kpis['Incident State Distribution'],
        'Incident_Location_Analysis': kpis['Incident Location Analysis'],
        'Average_Incident_Hour': f"{kpis['Average Incident Hour']}",
        'Policy_CSL_Distribution': kpis['Policy CSL Distribution'],
        'Hobbies_Analysis': kpis['Hobbies Analysis'],
        'Relationship_Status_Distribution': kpis['Relationship Status Distribution'],
        'Top_Incident_Cities': kpis['Top Incident Cities'],
        'Top_Incident_States': kpis['Top Incident States'],
        'Total_Capital_Gains': f"${kpis['Total Capital Gains']:,.2f}",
        'Total_Capital_Loss': f"${kpis['Total Capital Loss']:,.2f}",

    }
    
    print(kpis_formatted)

    return render(request, 'kpi_metrics.html', {'kpis': kpis_formatted, 'chart_data_1': chart_data_1, 'min_date': min_date, 'max_date': max_date})


# dashboard/views.py

from django.shortcuts import render
from .models import InsuranceClaim

def search_customer(request):
    if request.method == 'GET':
        policy_number = request.GET.get('policy_number', '').strip()
        
        if not policy_number:
            # If no policy_number is provided, show a message and an empty form
            return render(request, 'search_customer.html', {'error': 'No policy number provided.'})
        
        try:
            # Fetch the customer details based on the provided policy_number
            customer = InsuranceClaim.objects.get(policy_number=policy_number)
            return render(request, 'customer_detail.html', {'customer': customer})
        except InsuranceClaim.DoesNotExist:
            # If no customer matches, show a not found message
            return render(request, 'search_customer.html', {'error': 'No customer matches the given policy number.'})







from django.conf import settings
from django.shortcuts import render
from .models import InsuranceClaim
import pandas as pd
import joblib

# Load the saved model and scalers
model = joblib.load('voting_classifier_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_data(df, label_encoders, scaler):

    
    """Preprocess the data for prediction."""
    # Convert categorical columns to numeric using label encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Normalize numerical columns
    num_cols = ['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
                'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']
    df[num_cols] = scaler.transform(df[num_cols])

    # Calculate 'policy_tenure'
    df['policy_tenure'] = (pd.to_datetime(df['incident_date']) - pd.to_datetime(df['policy_bind_date'])).dt.days

    # Drop original date columns and 'incident_location'
    df.drop(columns=['policy_bind_date', 'incident_date', 'incident_location'], inplace=True)

    # Ensure only relevant features are included
    X_res = ['months_as_customer', 'age', 'policy_number', 'policy_state', 'policy_csl', 
             'policy_deductable', 'policy_annual_premium', 'insured_zip', 'insured_education_level', 
             'insured_occupation', 'insured_hobbies', 'insured_relationship', 'capital-gains', 
             'capital-loss', 'incident_type', 'collision_type', 'incident_severity', 
             'authorities_contacted', 'incident_state', 'incident_city', 'incident_hour_of_the_day', 
             'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'police_report_available', 
             'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 
             'auto_make', 'auto_model', 'auto_year', 'policy_tenure']
    df = df[X_res]

    return df

def predict_fraud(request):
    if request.method == 'POST':
        print(request.FILES)
        try:
            input_method = request.POST.get('input_method')
            print(f"Input Method: {input_method}")

            if input_method == 'file':
                # Handle File Upload
                uploaded_file = request.FILES.get('file')
                print(uploaded_file)
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)

                    except Exception as e:
                        print(f'Error reading CSV file: {e}')
                        return render(request, 'predict_fraud.html', {'error': f'Error processing the file: {e}'})

                    # Validate columns
                    required_columns = [
                        'months_as_customer', 'age', 'policy_number', 'policy_bind_date',
                        'policy_state', 'policy_csl', 'policy_deductable', 'policy_annual_premium',
                        'umbrella_limit', 'insured_zip', 'insured_sex', 'insured_education_level',
                        'insured_occupation', 'insured_hobbies', 'insured_relationship',
                        'capital-gains', 'capital-loss', 'incident_date', 'incident_type',
                        'collision_type', 'incident_severity', 'authorities_contacted',
                        'incident_state', 'incident_city', 'incident_location',
                        'incident_hour_of_the_day', 'number_of_vehicles_involved',
                        'property_damage', 'bodily_injuries', 'witnesses', 'police_report_available',
                        'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
                        'auto_make', 'auto_model', 'auto_year'
                    ]

                    missing_columns = set(required_columns) - set(df.columns)

                    if missing_columns:
                        error_message = f"Missing required columns: {', '.join(missing_columns)}"
                        return render(request, 'predict_fraud.html', {'error': error_message})

                    # Preprocess the DataFrame
                    preprocessed_df = preprocess_data(df, label_encoders, scaler)

                    # Make predictions
                    predictions = model.predict(preprocessed_df)
                    prediction_probs = model.predict_proba(preprocessed_df)

                    # Create results DataFrame
                    results_df = pd.DataFrame({
                        'PolicyNumber': df['policy_number'],
                        'Prediction': ['Fraud' if p == 1 else 'Not Fraud' for p in predictions],
                        'Probability': prediction_probs[:, 1]
                    })

                    print(results_df.head())
                    print(results_df.to_dict(orient='records'))

                    # Pass the results DataFrame to the template
                    return render(request, 'prediction_result.html', {'results': results_df.to_dict(orient='records')})

            elif input_method == 'manual':
                # Handle manual input
                user_input = {}

                def get_post_value(field, cast_type):
                    value = request.POST.get(field)
                    if value is None:
                        raise ValueError(f"Missing value for field: {field}")
                    try:
                        return cast_type(value)
                    except ValueError:
                        raise ValueError(f"Invalid value for field: {field}")

                user_input['months_as_customer'] = get_post_value('months_as_customer', int)
                user_input['age'] = get_post_value('age', int)
                user_input['policy_number'] = get_post_value('policy_number', int)
                user_input['policy_bind_date'] = request.POST.get('policy_bind_date')
                user_input['policy_state'] = request.POST.get('policy_state')
                user_input['policy_csl'] = request.POST.get('policy_csl')
                user_input['policy_deductable'] = get_post_value('policy_deductable', float)
                user_input['policy_annual_premium'] = get_post_value('policy_annual_premium', float)
                user_input['umbrella_limit'] = get_post_value('umbrella_limit', float)
                user_input['insured_zip'] = get_post_value('insured_zip', int)
                user_input['insured_sex'] = request.POST.get('insured_sex')
                user_input['insured_education_level'] = request.POST.get('insured_education_level')
                user_input['insured_occupation'] = request.POST.get('insured_occupation')
                user_input['insured_hobbies'] = request.POST.get('insured_hobbies')
                user_input['insured_relationship'] = request.POST.get('insured_relationship')
                user_input['capital-gains'] = get_post_value('capital-gains', float)
                user_input['capital-loss'] = get_post_value('capital-loss', float)
                user_input['incident_date'] = request.POST.get('incident_date')
                user_input['incident_type'] = request.POST.get('incident_type')
                user_input['collision_type'] = request.POST.get('collision_type')
                user_input['incident_severity'] = request.POST.get('incident_severity')
                user_input['authorities_contacted'] = request.POST.get('authorities_contacted')
                user_input['incident_state'] = request.POST.get('incident_state')
                user_input['incident_city'] = request.POST.get('incident_city')
                user_input['incident_location'] = request.POST.get('incident_location')
                user_input['incident_hour_of_the_day'] = get_post_value('incident_hour_of_the_day', int)
                user_input['number_of_vehicles_involved'] = get_post_value('number_of_vehicles_involved', int)
                user_input['property_damage'] = request.POST.get('property_damage')
                user_input['bodily_injuries'] = get_post_value('bodily_injuries', int)
                user_input['witnesses'] = get_post_value('witnesses', int)
                user_input['police_report_available'] = request.POST.get('police_report_available')
                user_input['total_claim_amount'] = get_post_value('total_claim_amount', float)
                user_input['injury_claim'] = get_post_value('injury_claim', float)
                user_input['property_claim'] = get_post_value('property_claim', float)
                user_input['vehicle_claim'] = get_post_value('vehicle_claim', float)
                user_input['auto_make'] = request.POST.get('auto_make')
                user_input['auto_model'] = request.POST.get('auto_model')
                user_input['auto_year'] = get_post_value('auto_year', int)

                print(user_input)

                # Preprocess input
                preprocessed_input = preprocess_data(pd.DataFrame([user_input]), label_encoders, scaler)
                print(preprocessed_input)
                
                # Make predictions
                prediction = model.predict(preprocessed_input)
                prediction_proba = model.predict_proba(preprocessed_input)

                result = {
                    'PolicyNumber': user_input['policy_number'],
                    'prediction': 'Fraud' if prediction[0] == 1 else 'Not Fraud',
                    'probability': prediction_proba[0][1]  # Probability of class 1 (Fraud)
                }

                print(result)
                
                return render(request, 'prediction_result.html', {'results': result})

        except ValueError as ve:
            # Handle conversion errors
            return render(request, 'predict_fraud.html', {'error': str(ve)})
        except Exception as e:
            # Handle any other exceptions
            return render(request, 'predict_fraud.html', {'error': str(e)})

    # For GET request, or if POST is not submitted
    # Fetch dropdown values for the form
    claims = InsuranceClaim.objects.all()
    data = pd.DataFrame(list(claims.values()))

    policy_states = data['policy_state'].dropna().unique()
    policy_csls = data['policy_csl'].dropna().unique()
    insured_education_levels = data['insured_education_level'].dropna().unique()
    insured_occupations = data['insured_occupation'].dropna().unique()
    insured_hobbies = data['insured_hobbies'].dropna().unique()
    insured_relationships = data['insured_relationship'].dropna().unique()
    incident_types = data['incident_type'].dropna().unique()
    collision_types = data['collision_type'].dropna().unique()
    incident_severities = data['incident_severity'].dropna().unique()
    authorities_contacted = data['authorities_contacted'].dropna().unique()
    incident_states = data['incident_state'].dropna().unique()
    property_damage_options = ['YES', 'NO']
    police_report_options = ['YES', 'NO']

    return render(request, 'predict_fraud.html', {
        'policy_states': policy_states,
        'policy_csls': policy_csls,
        'insured_education_levels': insured_education_levels,
        'insured_occupations': insured_occupations,
        'insured_hobbies': insured_hobbies,
        'insured_relationships': insured_relationships,
        'incident_types': incident_types,
        'collision_types': collision_types,
        'incident_severities': incident_severities,
        'authorities_contacted': authorities_contacted,
        'incident_states': incident_states,
        'property_damage_options': property_damage_options,
        'police_report_options': police_report_options
    })



