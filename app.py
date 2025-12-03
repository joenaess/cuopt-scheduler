import streamlit as st
import pandas as pd
import pulp
import random
import time
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker

# --- Configuration & Setup ---
st.set_page_config(page_title="NVIDIA cuOpt Scheduler", layout="wide")
fake = Faker()

# Check for cuOpt availability
try:
    from pulp import CUOPT
    CUOPT_AVAILABLE = True
except ImportError:
    CUOPT_AVAILABLE = False

# --- Helper Functions ---

def generate_mock_data(num_employees=10, num_days=7):
    """
    Generates realistic employee data with names, skills, and varying availability.
    """
    skills = ["RN (Registered Nurse)", "LPN (Lic. Practical Nurse)", "CNA (Cert. Nursing Asst)"]
    employee_data = []
    
    # Generate realistic employees
    for i in range(num_employees):
        name = fake.first_name() + " " + fake.last_name()
        skill = random.choice(skills)
        
        # Randomize Max Hours (Part-time vs Full-time logic)
        is_full_time = random.random() > 0.3
        max_hours = 40 if is_full_time else random.choice([24, 32])
        
        # Availability: 1 = Available, 0 = Unavailable
        # 3 shifts per day [Day, Evening, Night]
        availability = {day: [1, 1, 1] for day in range(num_days)}
        
        # Add some random unavailability (e.g., requests off)
        for day in range(num_days):
            if random.random() < 0.15: # 15% chance to request a day off
                availability[day] = [0, 0, 0]
            elif random.random() < 0.2: # 20% chance to be unavailable for specific shift
                availability[day][random.randint(0, 2)] = 0

        employee_data.append([name, skill, max_hours, availability])
    
    employee_df = pd.DataFrame(employee_data, columns=["Name", "Skill", "Max_Hours", "Availability"])
    employee_df.set_index("Name", inplace=True)
    
    # Generate Shift Requirements
    shift_data = []
    shift_names = ["Day", "Evening", "Night"]
    
    # Define staffing needs per shift (randomized slightly per day in a real scenario, but fixed here for simplicity)
    for shift_name in shift_names:
        # Higher demand during Day, lower at Night
        if shift_name == "Day":
            req = {"RN (Registered Nurse)": 2, "LPN (Lic. Practical Nurse)": 2, "CNA (Cert. Nursing Asst)": 3}
        elif shift_name == "Evening":
            req = {"RN (Registered Nurse)": 1, "LPN (Lic. Practical Nurse)": 2, "CNA (Cert. Nursing Asst)": 2}
        else:
            req = {"RN (Registered Nurse)": 1, "LPN (Lic. Practical Nurse)": 1, "CNA (Cert. Nursing Asst)": 1}
            
        shift_data.append([shift_name, req, 8])
    
    shift_df = pd.DataFrame(shift_data, columns=["shift_id", "required_staff", "shift_length"])
    shift_df.set_index("shift_id", inplace=True)
    
    return employee_df, shift_df

def solve_schedule(employee_df, shift_df, num_days, use_gpu=True):
    # Pre-processing
    shift_names = shift_df.index.tolist()
    employees = employee_df.index.tolist()
    
    model = pulp.LpProblem("Nurse_Scheduling_Optimization", pulp.LpMinimize)
    
    # Variables: x[employee, shift, day]
    x = pulp.LpVariable.dicts("assign",
                              [(emp, s, d) for emp in employees for s in shift_names for d in range(num_days)],
                              cat='Binary')

    # --- Objective ---
    # Minimize total shifts assigned (Efficiency) 
    # OR Maximize preference (not implemented in this mock, but possible)
    model += pulp.lpSum(x[emp, s, d] for emp in employees for s in shift_names for d in range(num_days)), "Minimize_Total_Shifts"

    # --- Constraints ---
    
    # 1. Staffing Demand: Ensure enough skilled people per shift
    for d in range(num_days):
        for s in shift_names:
            reqs = shift_df.loc[s, "required_staff"]
            for skill, count in reqs.items():
                model += pulp.lpSum(x[emp, s, d] 
                                    for emp in employees 
                                    if employee_df.loc[emp, "Skill"] == skill) >= count

    # 2. Max Hours per Employee
    for emp in employees:
        model += pulp.lpSum(x[emp, s, d] * shift_df.loc[s, "shift_length"] 
                            for s in shift_names 
                            for d in range(num_days)) <= employee_df.loc[emp, "Max_Hours"]

    # 3. No Overlapping Shifts (1 shift per day)
    for emp in employees:
        for d in range(num_days):
            model += pulp.lpSum(x[emp, s, d] for s in shift_names) <= 1

    # 4. Respect Availability
    for emp in employees:
        avail = employee_df.loc[emp, "Availability"]
        for d in range(num_days):
            for i, s in enumerate(shift_names):
                if avail[d][i] == 0:
                    model += x[emp, s, d] == 0

    # --- Solve ---
    status_msg = "Solved with CPU (CBC)"
    if use_gpu:
        try:
            if CUOPT_AVAILABLE:
                solver = pulp.CUOPT(timeLimit=30)
                model.solve(solver)
                status_msg = "🚀 Solved with NVIDIA cuOpt"
            else:
                st.warning("pulp.CUOPT not detected. Using default solver.")
                model.solve()
        except Exception as e:
            st.error(f"Solver Error: {e}")
            model.solve()
    else:
        model.solve()

    # --- Post-Processing ---
    if pulp.LpStatus[model.status] == 'Optimal':
        results = []
        day_map = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
        
        for d in range(num_days):
            for s in shift_names:
                for emp in employees:
                    if x[emp, s, d].varValue > 0.9:
                        results.append({
                            "Day_Index": d,
                            "Day": f"Day {d+1} ({day_map.get(d%7, '')})",
                            "Shift": s,
                            "Employee": emp,
                            "Skill": employee_df.loc[emp, "Skill"],
                            "Start": d + (0.33 if s=="Day" else 0.66 if s=="Evening" else 0.99), # Fake timestamp for Gantt
                            "Duration": 0.33
                        })
        return pd.DataFrame(results), status_msg
    return None, "Infeasible"

# --- UI Layout ---

st.title("🏥 Intelligent Staff Scheduler")
st.caption("Powered by NVIDIA cuOpt & Streamlit")

with st.sidebar:
    st.header("1. Data Generation")
    num_emp = st.slider("Number of Employees", 10, 100, 25)
    duration = st.slider("Schedule Duration (Days)", 7, 21, 7)
    
    st.divider()
    
    st.header("2. Optimization Settings")
    use_gpu = st.toggle("Use NVIDIA GPU (cuOpt)", value=True)
    
    st.info("""
    **What can I alter in this simulation?**
    
    In a real deployment, you can adjust:
    * **Staffing Rules:** Min/Max staff per shift.
    * **Skills:** Required certifications (RN vs CNA).
    * **Constraints:** Max overtime, union rules, consecutive work days.
    * **Objectives:** Minimize cost vs. Maximize employee preference.
    """)
    
    if st.button("Generate & Solve", type="primary"):
        run_solver = True
    else:
        run_solver = False

# --- Main Execution ---

if run_solver:
    # 1. Generate Data
    emp_df, shift_df = generate_mock_data(num_emp, duration)
    
    # 2. Solve
    start_t = time.time()
    with st.spinner("Optimizing schedule..."):
        schedule_df, status = solve_schedule(emp_df, shift_df, duration, use_gpu)
    end_t = time.time()
    
    if schedule_df is not None:
        st.success(f"{status} in {end_t - start_t:.4f} seconds!")
        
        # --- Visualizations ---
        
        # Tabbed View
        tab1, tab2, tab3 = st.tabs(["📅 Visual Schedule (Gantt)", "📊 Coverage Analysis", "📝 Detailed List"])
        
        with tab1:
            st.subheader("Employee Schedule Timeline")
            # Create a nice Gantt chart using Plotly
            # We map shifts to specific hours for visualization purposes
            shift_hour_map = {"Day": "07:00", "Evening": "15:00", "Night": "23:00"}
            schedule_df['Start_Time'] = schedule_df.apply(lambda x: f"2023-01-{x['Day_Index']+1:02d} {shift_hour_map[x['Shift']]}", axis=1)
            schedule_df['End_Time'] = pd.to_datetime(schedule_df['Start_Time']) + pd.Timedelta(hours=8)
            
            fig_gantt = px.timeline(schedule_df, 
                                    x_start="Start_Time", 
                                    x_end="End_Time", 
                                    y="Employee", 
                                    color="Shift",
                                    hover_data=["Skill", "Day"],
                                    title="Staff Shifts Overview",
                                    color_discrete_map={"Day": "#00CC96", "Evening": "#AB63FA", "Night": "#19D3F3"})
            fig_gantt.update_yaxes(autorange="reversed") # Name order top to bottom
            st.plotly_chart(fig_gantt, use_container_width=True)

        with tab2:
            st.subheader("Shift Coverage Heatmap")
            # Pivot to see count of staff per shift/day
            coverage = schedule_df.pivot_table(index="Shift", columns="Day", values="Employee", aggfunc='count')
            
            fig_heat = px.imshow(coverage, 
                                 text_auto=True, 
                                 aspect="auto",
                                 color_continuous_scale="Viridis",
                                 title="Number of Staff Assigned per Shift")
            st.plotly_chart(fig_heat, use_container_width=True)
            
            st.markdown("**Skill Breakdown:**")
            skill_counts = schedule_df['Skill'].value_counts()
            st.bar_chart(skill_counts)

        with tab3:
            st.dataframe(schedule_df[["Day", "Shift", "Employee", "Skill"]], use_container_width=True)
            
            # Download
            csv = schedule_df.to_csv(index=False)
            st.download_button("Download Schedule CSV", csv, "optimized_schedule.csv", "text/csv")
            
    else:
        st.error("Optimization failed! The constraints might be too tight (e.g., not enough staff for the required shifts). Try increasing the employee count.")
        
else:
    st.info("👈 Adjust settings in the sidebar and click 'Generate & Solve' to start.")
    
    # Show example data structure to user initially
    st.markdown("### Preview of Realistic Input Data")
    dummy_emp, dummy_shift = generate_mock_data(5, 7)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Employee Pool (Sample)**")
        st.dataframe(dummy_emp[["Skill", "Max_Hours"]].head())
    with col2:
        st.write("**Shift Requirements**")
        st.dataframe(dummy_shift[["required_staff"]])