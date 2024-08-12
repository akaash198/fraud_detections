import pandas as pd
import pickle
import logging
from django.core.management.base import BaseCommand
from dashboard.models import InsuranceClaim
from django.db import transaction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Load insurance claims data and machine learning model'

    def handle(self, *args, **kwargs):
        try:
            # Load the model
            with open('dashboard/management/commands/voting_classifier_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            logger.info('Model loaded successfully')
        except FileNotFoundError:
            logger.error('Model file not found')
            return
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            return
        
        try:
            # Load and preprocess data
            df = pd.read_csv('dashboard/management/commands/insurance_claims_data.csv')
            logger.info('Data loaded successfully')
            
            # Convert relevant columns to datetime format
            df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
            df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')

            with transaction.atomic():
                # Clear existing data
                InsuranceClaim.objects.all().delete()
                for _, row in df.iterrows():
                    InsuranceClaim.objects.create(
                        months_as_customer=row['months_as_customer'],
                        age=row['age'],
                        policy_number=row['policy_number'],
                        policy_bind_date=row['policy_bind_date'],
                        policy_state=row['policy_state'],
                        policy_csl=row['policy_csl'],
                        policy_deductable=row['policy_deductable'],
                        policy_annual_premium=row['policy_annual_premium'],
                        umbrella_limit=row['umbrella_limit'],
                        insured_zip=row['insured_zip'],
                        insured_sex=row['insured_sex'],
                        insured_education_level=row['insured_education_level'],
                        insured_occupation=row['insured_occupation'],
                        insured_hobbies=row['insured_hobbies'],
                        insured_relationship=row['insured_relationship'],
                        capital_gains=row['capital-gains'],
                        capital_loss=row['capital-loss'],
                        incident_date=row['incident_date'],
                        incident_type=row['incident_type'],
                        collision_type=row['collision_type'],
                        incident_severity=row['incident_severity'],
                        authorities_contacted=row['authorities_contacted'],
                        incident_state=row['incident_state'],
                        incident_city=row['incident_city'],
                        incident_location=row['incident_location'],
                        incident_hour_of_the_day=row['incident_hour_of_the_day'],
                        number_of_vehicles_involved=row['number_of_vehicles_involved'],
                        property_damage=row['property_damage'],
                        bodily_injuries=row['bodily_injuries'],
                        witnesses=row['witnesses'],
                        police_report_available=row['police_report_available'],
                        total_claim_amount=row['total_claim_amount'],
                        injury_claim=row['injury_claim'],
                        property_claim=row['property_claim'],
                        vehicle_claim=row['vehicle_claim'],
                        auto_make=row['auto_make'],
                        auto_model=row['auto_model'],
                        auto_year=row['auto_year'],
                        fraud_reported=row['fraud_reported']
                    )
            self.stdout.write(self.style.SUCCESS('Data loaded successfully'))
        except pd.errors.EmptyDataError:
            logger.error('No data found in the CSV file')
        except pd.errors.ParserError:
            logger.error('Error parsing the CSV file')
        except Exception as e:
            logger.error(f'Error processing data: {e}')
