from django.contrib import admin
from .models import InsuranceClaim

class InsuranceClaimAdmin(admin.ModelAdmin):
    list_display = (
        'months_as_customer',
        'age',
        'policy_number',
        'policy_bind_date',
        'policy_state',
        'policy_csl',
        'policy_deductable',
        'policy_annual_premium',
        'umbrella_limit',
        'insured_zip',
        'insured_sex',
        'insured_education_level',
        'insured_occupation',
        'insured_hobbies',
        'insured_relationship',
        'capital_gains',
        'capital_loss',
        'incident_date',
        'incident_type',
        'collision_type',
        'incident_severity',
        'authorities_contacted',
        'incident_state',
        'incident_city',
        'incident_location',
        'incident_hour_of_the_day',
        'number_of_vehicles_involved',
        'property_damage',
        'bodily_injuries',
        'witnesses',
        'police_report_available',
        'total_claim_amount',
        'injury_claim',
        'property_claim',
        'vehicle_claim',
        'auto_make',
        'auto_model',
        'auto_year',
        'fraud_reported'
    )
    
    list_filter = (
        'fraud_reported',
        'incident_type',
        'collision_type',
        'incident_severity',
        'authorities_contacted',
        'incident_state',
        'policy_state',
        'insured_sex'
    )
    
    search_fields = (
        'policy_number',
        'insured_zip',
        'auto_make',
        'auto_model',
        'incident_location',
        'policy_csl'
    )

admin.site.register(InsuranceClaim, InsuranceClaimAdmin)
