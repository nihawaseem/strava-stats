import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from urllib.parse import urlencode
import numpy as np
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv


# Strava API configuration
load_dotenv()
STRAVA_CLIENT_ID = os.getenv('STRAVA_CLIENT_ID', 'your_client_id_here')
STRAVA_CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET', 'your_client_secret_here')
REDIRECT_URI = os.getenv('REDIRECT_URI', 'https://stravastats.streamlit.app')

# Page config
st.set_page_config(
    page_title="strava stats",
    page_icon="💃",
    layout="wide"
)

#  Generate Strava OAuth authorization URL
def get_authorization_url():
    params = {
        'client_id': STRAVA_CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'approval_prompt': 'force',
        'scope': 'read,activity:read_all'
    }
    return f"https://www.strava.com/oauth/authorize?{urlencode(params)}"

#  Exchange authorization code for access token
def exchange_code_for_token(code):
    data = {
        'client_id': STRAVA_CLIENT_ID,
        'client_secret': STRAVA_CLIENT_SECRET,
        'code': code,
        'grant_type': 'authorization_code'
    }
    
    response = requests.post('https://www.strava.com/oauth/token', data=data)
    if response.status_code == 200:
        return response.json()
    return None

#  Refresh expired access token
def refresh_access_token(refresh_token):
    data = {
        'client_id': STRAVA_CLIENT_ID,
        'client_secret': STRAVA_CLIENT_SECRET,
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token'
    }
    
    response = requests.post('https://www.strava.com/oauth/token', data=data)
    if response.status_code == 200:
        return response.json()
    return None

#  Fetch activities from Strava API
def get_activities(access_token, per_page=30, page=1):
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {
        'per_page': per_page,

        'page': page
    }
    
    response = requests.get(
        'https://www.strava.com/api/v3/athlete/activities',
        headers=headers,
        params=params
    )
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        st.error("Access token expired. Please re-authenticate.")
        return None
    else:
        st.error(f"Error fetching activities: {response.status_code}")
        return None

#  Convert activities to pandas DataFrame and clean data
def process_activities_data(activities):
    if not activities:
        return pd.DataFrame()
    
    df = pd.DataFrame(activities)
    
    # Convert date columns
    df['start_date_local'] = pd.to_datetime(df['start_date_local'])
    df['date'] = df['start_date_local'].dt.date
    
    # Convert distance from meters to km
    df['distance_km'] = df['distance'] / 1000
    
    # Convert moving time to hours
    df['moving_time_hours'] = df['moving_time'] / 3600
    
    # Calculate pace (min/km) for running activities
    df['pace_min_per_km'] = df.apply(
        lambda row: (row['moving_time'] / 60) / (row['distance'] / 1000) 
        if row['distance'] > 0 else 0, axis=1
    )
    
    # Convert elevation to meters (it's already in meters)
    df['elevation_gain_m'] = df['total_elevation_gain']
    
    return df

#  Create summary metrics for the dashboard
def create_summary_metrics(df):
    if df.empty:
        return {}
    
    total_activities = len(df)
    total_distance = df['distance_km'].sum()
    total_time = df['moving_time_hours'].sum()
    avg_distance = df['distance_km'].mean()
    
    return {
        'total_activities': total_activities,
        'total_distance_km': round(total_distance, 1),
        'total_time_hours': round(total_time, 1),
        'avg_distance_km': round(avg_distance, 1)
    }

def main():
    st.title("💃 strava stats")
    
    # Check if we have stored tokens
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
        st.session_state.refresh_token = None
    
    # Handle OAuth callback
    query_params = st.query_params
    if 'code' in query_params and st.session_state.access_token is None:
        code = query_params['code']
        token_data = exchange_code_for_token(code)
        
        if token_data:
            st.session_state.access_token = token_data['access_token']
            st.session_state.refresh_token = token_data['refresh_token']
            st.success("Successfully authenticated with Strava!")
            st.rerun()
        else:
            st.error("Failed to authenticate with Strava")
    
    # Authentication section
    if not st.session_state.access_token:
        st.header("🔐 Authentication Required")
        st.write("To view your Strava data, you need to authenticate with Strava.")
        
        st.write("**Setup Instructions:**")
        st.write("1. Go to https://www.strava.com/settings/api")
        st.write("2. Create a new application")
        st.write("3. Set Authorization Callback Domain to: `stravastats.streamlit.io`")
        st.write("4. Add your Client ID and Client Secret to environment variables")
        
        if st.button("🚀 Connect to Strava", type="primary"):
            auth_url = get_authorization_url()
            st.write("Click the link below to authorize this application:")
            st.link_button("Authorize with Strava", auth_url)
        
        return
    
    # Main dashboard
    st.header("doing statistics with strava")
    
    # Fetch activities
    with st.spinner("Fetching your activities..."):
        activities = get_activities(st.session_state.access_token, per_page=50)
    
    if not activities:
        if st.button("🔄 Refresh Token"):
            if st.session_state.refresh_token:
                token_data = refresh_access_token(st.session_state.refresh_token)
                if token_data:
                    st.session_state.access_token = token_data['access_token']
                    st.session_state.refresh_token = token_data['refresh_token']
                    st.rerun()
        return
    
    # Process data
    df = process_activities_data(activities)
    
    if df.empty:
        st.warning("No activities found!")
        return
    
    # Summary metrics
    metrics = create_summary_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Activities", metrics['total_activities'])
    with col2:
        st.metric("Total Distance", f"{metrics['total_distance_km']} km")
    with col3:
        st.metric("Total Time", f"{metrics['total_time_hours']} hrs")
    with col4:
        st.metric("Avg Distance", f"{metrics['avg_distance_km']} km")
    
    # Activity type filter
    activity_types = df['sport_type'].unique()
    selected_types = st.multiselect(
        "Filter by Activity Type",
        activity_types,
        default=activity_types
    )
    
    filtered_df = df[df['sport_type'].isin(selected_types)]
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Distance Over Time")
        fig_distance = px.line(
            filtered_df.sort_values('start_date_local'),
            x='start_date_local',
            y='distance_km',
            color='sport_type',
            title="Distance per Activity"
        )
        fig_distance.update_layout(height=400)
        st.plotly_chart(fig_distance, use_container_width=True)
    
    with col2:
        st.subheader("🏃‍♂️ Activities by Type")
        activity_counts = filtered_df['sport_type'].value_counts()
        fig_pie = px.pie(
            values=activity_counts.values,
            names=activity_counts.index,
            title="Activity Distribution"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Weekly summary
    st.subheader("📅 Weekly Summary")
    filtered_df['week'] = filtered_df['start_date_local'].dt.to_period('W')
    weekly_summary = filtered_df.groupby('week').agg({
        'distance_km': 'sum',
        'moving_time_hours': 'sum',
        'name': 'count'
    }).reset_index()
    weekly_summary.columns = ['Week', 'Total Distance (km)', 'Total Time (hrs)', 'Activities']
    weekly_summary['Week'] = weekly_summary['Week'].astype(str)
    
    fig_weekly = px.bar(
        weekly_summary.tail(8),  # Last 8 weeks
        x='Week',
        y='Total Distance (km)',
        title="Weekly Distance Summary"
    )
    fig_weekly.update_layout(height=400)
    st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Recent activities table
    st.subheader("🕐 Recent Activities")
    recent_activities = filtered_df.head(10)[
        ['name', 'sport_type', 'date', 'distance_km', 'moving_time_hours', 'elevation_gain_m']
    ].copy()
    recent_activities.columns = ['Activity', 'Type', 'Date', 'Distance (km)', 'Time (hrs)', 'Elevation (m)']
    recent_activities = recent_activities.round(2)
    st.dataframe(recent_activities, use_container_width=True)

    # Regression
    st.subheader("🕐 Predicted pace")
    recent_activities = filtered_df.head(10)[
        ['date', 'distance_km', 'moving_time_hours']
    ].copy()
    recent_activities['pace'] = recent_activities['moving_time_hours'] / recent_activities['distance_km']
    recent_activities['date_ordinal'] = pd.to_datetime(recent_activities['date']).map(pd.Timestamp.toordinal)
    # Fit linear regression model
    X = recent_activities[['date_ordinal']]
    y = recent_activities['pace']
    model = LinearRegression()
    model.fit(X, y)

    # Add predicted values
    recent_activities['predicted_pace'] = model.predict(X)

    # Plot actual vs predicted
    pred_pace = px.line(
            recent_activities.sort_values('date'),
            x='date',
            y='predicted_pace',
            title="Regression fit to pace over time"
        )
    
    pred_pace.add_trace(
    go.Scatter(
        x=recent_activities['date'],
        y=recent_activities['pace'],
        mode='markers',
        name='Actual Pace',
        marker=dict(color='blue', size=6)
    )
)
    
    pred_pace.update_layout(height=400)
    st.plotly_chart(pred_pace, use_container_width=True)
    st.plotly_chart(pred_pace, use_container_width=True)
    
    # Reset authentication
    if st.button("🚪 Logout"):
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.rerun()

if __name__ == "__main__":
    main()
