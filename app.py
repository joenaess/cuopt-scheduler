import streamlit as st
import pandas as pd
import pulp
import random
import os
import time

# Check if cuOpt is available directly in PuLP, otherwise define a wrapper
try:
    from pulp import CUOPT
    CUOPT_AVAILABLE = True
except ImportError:
    CUOPT_AVAILABLE = False
    # If standard pulp doesn't have CUOPT yet, we can use the cuopt_sh_client manually
    # or rely on standard solvers if cuopt fails. 
    # For this tutorial, we will focus on the code structure assuming standard usage
    # or providing a clear path to use the solver.

# --- Function to Generate Mock Data (Same as before) ---
def generate_mock_data(num_employees=10, num_shifts=3, num_days=7):
    skills = ["RN", "LPN", "CNA"]
    employee_data = []
    for i in range(num_employees):
        employee_id = f"E{i+1}"
        skill = random.choice(skills)
        max_hours = random.randint(30, 48)
        # Random availability
        availability = {day: [1] * num_shifts for day in range(num_days)}
        for day in range(num_days):
            for shift in range(num_shifts):
                if random.random() < 0.2:
                    availability[day][shift] = 0
        employee_data.append([employee_id, skill, max_hours, availability])
    
    employee_df = pd.DataFrame(employee_data, columns=["employee_id", "skill", "max_hours", "availability"])
    employee_df.set_index("employee_id", inplace=True)
    employee_df['availability'] = employee_df['availability'].apply(lambda x: str(x))

    shift_data = []
    shift_names = ["Day", "Evening", "Night"]
    for shift_name in shift_names:
        required_staff = {skill: random.randint(0, 2) for skill in skills}
        shift_data.append([shift_name, required_staff, 8])
    
    shift_df = pd.DataFrame(shift_data, columns=["shift_id", "required_staff", "shift_length"])
    shift_df.set_index("shift_id", inplace=True)
    shift_df['required_staff'] = shift_df['required_staff'].apply(lambda x: str(x))
    return employee_df, shift_df

# --- Function to Solve using NVIDIA cuOpt ---
def solve_schedule(employee_df, shift_df, use_gpu=True):
    # Data Preprocessing
    employee_df = employee_df.copy()
    shift_df = shift_df.copy()
    employee_df['availability'] = employee_df['availability'].apply(eval)
    shift_df['required_staff'] = shift_df['required_staff'].apply(eval)

    num_days = len(employee_df['availability'].iloc[0])
    shifts = shift_df.index.tolist()
    employees = employee_df.index.tolist()

    # Initialize Model
    model = pulp.LpProblem("Nursing_Home_Scheduling", pulp.LpMinimize)
    
    # Variables
    x = pulp.LpVariable.dicts("schedule",
                              [(employee, shift, day)
                               for employee in employees
                               for shift in shifts
                               for day in range(num_days)],
                              cat='Binary')

    # Objective: Minimize total shifts
    model += pulp.lpSum(x[employee, shift, day]
                        for employee in employees
                        for shift in shifts
                        for day in range(num_days)), "Total_Shifts"

    # Constraints
    # 1. Staffing Requirements
    for day in range(num_days):
        for shift in shifts:
            for skill, required in shift_df.loc[shift, "required_staff"].items():
                model += pulp.lpSum(x[employee, shift, day]
                                    for employee in employees
                                    if employee_df.loc[employee, "skill"] == skill) >= required

    # 2. One shift per day per employee
    for employee in employees:
        for day in range(num_days):
            model += pulp.lpSum(x[employee, shift, day] for shift in shifts) <= 1

    # 3. Max hours
    for employee in employees:
        model += pulp.lpSum(x[employee, shift, day] * shift_df.loc[shift, "shift_length"]
                            for shift in shifts
                            for day in range(num_days)) <= employee_df.loc[employee, "max_hours"]

    # 4. Availability
    for employee in employees:
        for day in range(num_days):
            for shift_index, shift in enumerate(shifts):
                if employee_df.loc[employee, "availability"][day][shift_index] == 0:
                    model += x[employee, shift, day] == 0

    # --- SOLVER SELECTION ---
    if use_gpu:
        try:
            # OPTION 1: Direct Integration (Preferred if available in your lib version)
            if hasattr(pulp, 'CUOPT'):
                solver = pulp.CUOPT(timeLimit=30)
                st.info(f"🚀 Solving with NVIDIA cuOpt (Direct Integration)...")
                model.solve(solver)
            
            # OPTION 2: Fallback (if pulp.CUOPT is missing)
            # This uses the default solver but prints a warning. 
            # Real cuOpt manual integration requires writing an MPS file and passing it to the cuopt_sh_client.
            else:
                st.warning("⚠️ pulp.CUOPT not found. Falling back to default CPU solver (CBC). Update PuLP or check NVIDIA docs.")
                model.solve()
        except Exception as e:
            st.error(f"GPU Solve failed: {e}. Falling back to CPU.")
            model.solve()
    else:
        st.info("Solving with CPU (CBC)...")
        model.solve()

    # Process Results
    if pulp.LpStatus[model.status] == 'Optimal':
        schedule_data = []
        for day in range(num_days):
            for shift in shifts:
                for employee in employees:
                    if x[employee, shift, day].varValue > 0.9:
                        schedule_data.append([day, shift, employee, employee_df.loc[employee, 'skill']])
        
        res_df = pd.DataFrame(schedule_data, columns=['Day', 'Shift', 'Employee', 'Skill'])
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon2", "Tue2"] # Extended list
        # Safe map
        res_df['Day_Name'] = res_df['Day'].apply(lambda d: day_names[d % 7])
        return res_df, "Optimal"
    else:
        return None, pulp.LpStatus[model.status]

# --- Streamlit UI ---
st.title("⚡ NVIDIA cuOpt Scheduler")
st.markdown("### Powered by RTX 4080 Laptop GPU")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    num_employees = st.slider("Employees", 10, 500, 50) # Increased max for GPU demo
    num_days = st.slider("Schedule Duration (Days)", 7, 30, 7)
    use_gpu = st.toggle("Enable GPU Acceleration", value=True)

# Generate Data
employee_df, shift_df = generate_mock_data(num_employees, 3, num_days)

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Employees:** {len(employee_df)}")
    st.dataframe(employee_df.head(), height=150)
with col2:
    st.write(f"**Shifts:** {len(shift_df)}")
    st.dataframe(shift_df, height=150)

if st.button("Run Optimization"):
    start_time = time.time()
    
    with st.spinner("Optimizing schedule..."):
        schedule, status = solve_schedule(employee_df, shift_df, use_gpu=use_gpu)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if status == "Optimal":
        st.success(f"✅ Found Optimal Schedule in {duration:.4f} seconds")
        st.dataframe(schedule)
        
        csv = schedule.to_csv(index=False)
        st.download_button("Download CSV", csv, "cuopt_schedule.csv", "text/csv")
    else:
        st.error(f"Optimization failed: {status}")