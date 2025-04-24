import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import List, Dict, Tuple, Optional
import yaml
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from dataclasses import dataclass
import random
import math


# ---------- Data Structures ----------

@dataclass
class Student:
    """Represents a student with preferences and attributes for school matching."""
    name: str
    preferences: List[str]  # Ordered list of preferred schools
    score: int  # Academic score (0-100)
    city: str  # Home city (used for local priority)
    match: Optional['School'] = None  # Assigned school


@dataclass
class School:
    """Represents a school with capacity and matching constraints."""
    name: str
    capacity: int  # Maximum number of students
    city: str  # Location (used for local priority)
    matches: List[Tuple[int, int, Student]] = None  # (score, priority, student)
    applicants: List[Student] = None  # All students who applied

    def __post_init__(self):
        self.matches = []
        self.applicants = []

    def prefers(self, student: Student, city_bonus: int) -> Tuple[bool, Optional[Student]]:
        """
        Determine if school prefers this student over current matches.
        Returns (accepted, rejected_student) tuple.
        """
        self.applicants.append(student)
        student_priority = city_bonus if student.city == self.city else 0

        if len(self.matches) < self.capacity:
            return True, None  # Accept if there's space

        # Sort matches by total score (score + priority)
        self.matches.sort(key=lambda x: (x[0] + x[1]))

        # Compare with the worst current match
        worst = self.matches[0]
        if (student.score + student_priority) > (worst[0] + worst[1]):
            return True, worst[2]  # Accept and reject the worst
        return False, None  # Reject

    def add_student(self, student: Student, city_bonus: int):
        """Add a student to the school's matches."""
        student_priority = city_bonus if student.city == self.city else 0
        self.matches.append((student.score, student_priority, student))
        student.match = self

    def remove_student(self, student: Student):
        """Remove a student from the school's matches."""
        self.matches = [t for t in self.matches if t[2] != student]
        if student.match == self:
            student.match = None


# ---------- Matching Algorithms ----------

def gale_shapley(students: List[Student], schools: Dict[str, School], city_bonus: int):
    """
    Implement the Gale-Shapley algorithm for stable matching between students and schools.

    Args:
        students: List of all student objects
        schools: Dictionary of school name to School object
        city_bonus: Priority points given to local students
    """
    free_students = [s for s in students if s.match is None]

    while free_students:
        student = free_students.pop(0)

        for school_name in student.preferences:
            school = schools[school_name]
            accepted, rejected_student = school.prefers(student, city_bonus)

            if accepted:
                if rejected_student:
                    # Handle rejection
                    rejected_student.match = None
                    free_students.append(rejected_student)
                    school.remove_student(rejected_student)

                if student.match:
                    # Remove from previous match if any
                    student.match.remove_student(student)

                # Add to new match
                school.add_student(student, city_bonus)
                break


def random_matching(students: List[Student], schools: Dict[str, School]):
    """
    Random matching algorithm for comparison with Gale-Shapley.
    This demonstrates the importance of the stable matching algorithm.
    """
    school_list = list(schools.values())

    for student in students:
        if student.match:
            student.match.remove_student(student)
        student.match = None

    for school in school_list:
        school.matches = []
        school.applicants = []

    # Shuffle students and assign randomly
    shuffled_students = students.copy()
    random.shuffle(shuffled_students)

    for school in school_list:
        while len(school.matches) < school.capacity and shuffled_students:
            student = shuffled_students.pop()
            school.add_student(student, 0)


# ---------- Analysis and Reporting ----------

def calculate_satisfaction(students: List[Student], schools: Dict[str, School]) -> Dict[str, float]:
    """
    Calculate satisfaction metrics for students and schools.
    Returns dictionary with various satisfaction measures.
    """
    # Student satisfaction
    student_satisfaction = {"1st": 0, "2nd": 0, "3rd": 0, "Unmatched": 0}
    total_preference_rank = 0
    matched_students = 0

    for student in students:
        if student.match:
            try:
                rank = student.preferences.index(student.match.name) + 1
                total_preference_rank += rank
                matched_students += 1

                if rank == 1:
                    student_satisfaction["1st"] += 1
                elif rank == 2:
                    student_satisfaction["2nd"] += 1
                else:
                    student_satisfaction["3rd"] += 1
            except ValueError:
                student_satisfaction["Unmatched"] += 1
        else:
            student_satisfaction["Unmatched"] += 1

    # School satisfaction (how full they are)
    school_fill_rate = sum(len(s.matches) for s in schools.values()) / sum(s.capacity for s in schools.values())

    # Average score of matched students
    matched_scores = [s.score for s in students if s.match]
    avg_score = sum(matched_scores) / len(matched_scores) if matched_scores else 0

    return {
        "student_satisfaction": student_satisfaction,
        "avg_preference_rank": total_preference_rank / matched_students if matched_students else 0,
        "school_fill_rate": school_fill_rate,
        "match_rate": matched_students / len(students),
        "avg_score": avg_score
    }


def generate_report(students: List[Student], schools: Dict[str, School]) -> str:
    """Generate detailed text report of matching results."""
    report = []

    # Student matches
    report.append("STUDENT MATCHES:")
    for student in students:
        if student.match:
            try:
                pref_rank = student.preferences.index(student.match.name) + 1
                report.append(f"{student.name}: {student.match.name} (Choice #{pref_rank})")
            except ValueError:
                report.append(f"{student.name}: {student.match.name} (Not in preferences)")
        else:
            report.append(f"{student.name}: Unmatched")

    # School statistics
    report.append("\nSCHOOL STATISTICS:")
    for school in schools.values():
        fill_rate = len(school.matches) / school.capacity
        avg_score = sum(s[0] for s in school.matches) / len(school.matches) if school.matches else 0
        report.append(
            f"{school.name}: {len(school.matches)}/{school.capacity} "
            f"({fill_rate:.0%}), Avg score: {avg_score:.1f}"
        )

    # Summary statistics
    satisfaction = calculate_satisfaction(students, schools)
    report.append("\nSUMMARY STATISTICS:")
    report.append(f"Students matched to 1st choice: {satisfaction['student_satisfaction']['1st']}")
    report.append(f"Students matched to 2nd choice: {satisfaction['student_satisfaction']['2nd']}")
    report.append(f"Students matched to 3rd choice: {satisfaction['student_satisfaction']['3rd']}")
    report.append(f"Unmatched students: {satisfaction['student_satisfaction']['Unmatched']}")
    report.append(f"Average preference rank: {satisfaction['avg_preference_rank']:.2f}")
    report.append(f"School fill rate: {satisfaction['school_fill_rate']:.0%}")
    report.append(f"Average score of matched students: {satisfaction['avg_score']:.1f}")

    return "\n".join(report)


# ---------- Visualization ----------

def plot_satisfaction(satisfaction: Dict[str, int], title: str = "Student Satisfaction"):
    """Create bar plot of student satisfaction distribution."""
    plt.figure(figsize=(10, 6))
    categories = ["1st Choice", "2nd Choice", "3rd Choice", "Unmatched"]
    values = [
        satisfaction["1st"],
        satisfaction["2nd"],
        satisfaction["3rd"],
        satisfaction["Unmatched"]
    ]
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']  # Green, Light Green, Yellow, Red

    bars = plt.bar(categories, values, color=colors)
    plt.title(title, fontsize=14)
    plt.xlabel("Match Quality", fontsize=12)
    plt.ylabel("Number of Students", fontsize=12)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()


def plot_matching_graph(students: List[Student], schools: Dict[str, School]):
    """Create bipartite graph visualization of matches."""
    G = nx.Graph()

    # Add nodes
    student_nodes = [(s.name, {"type": "student", "score": s.score}) for s in students]
    school_nodes = [(s.name, {"type": "school", "capacity": s.capacity}) for s in schools.values()]

    G.add_nodes_from(student_nodes, bipartite=0)
    G.add_nodes_from(school_nodes, bipartite=1)

    # Add edges for matches
    for school in schools.values():
        for _, _, student in school.matches:
            priority = student.city == school.city
            G.add_edge(student.name, school.name,
                       weight=2 if priority else 1,
                       style='solid',
                       color='blue' if priority else 'gray')

    # Add edges for applications without matches
    for school in schools.values():
        matched_names = {s[2].name for s in school.matches}
        for student in school.applicants:
            if student.name not in matched_names:
                G.add_edge(student.name, school.name,
                           weight=0.5,
                           style='dashed',
                           color='red')

    # Position nodes
    pos = {}
    student_y = 1.0
    school_y = 1.0

    for i, student in enumerate(students):
        pos[student.name] = (0, student_y - i * 0.8)

    for i, school in enumerate(schools.values()):
        pos[school.name] = (1, school_y - i * 0.8)

    # Draw
    plt.figure(figsize=(12, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[s.name for s in students],
                           node_color='skyblue', node_size=800, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=[s.name for s in schools.values()],
                           node_color='lightgreen', node_size=1000, alpha=0.8)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Draw edges
    solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d['style'] == 'solid']
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d['style'] == 'dashed']

    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=1.5,
                           edge_color=[d['color'] for _, _, d in G.edges(data=True) if d['style'] == 'solid'])
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, width=1, style='dashed',
                           edge_color='red', alpha=0.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Student',
               markerfacecolor='skyblue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='School',
               markerfacecolor='lightgreen', markersize=10),
        Line2D([0], [0], color='blue', lw=2, label='Priority Match'),
        Line2D([0], [0], color='gray', lw=2, label='Regular Match'),
        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Unmatched Application')
    ]

    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1),
               ncol=3, fontsize=10)

    plt.title("College Admissions Matching", fontsize=14)
    plt.axis('off')
    plt.tight_layout()


def plot_parameter_analysis(results: List[Dict[str, float]], param_name: str):
    """Plot how different parameter values affect matching outcomes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Impact of {param_name} on Matching Outcomes", fontsize=14)

    # Extract data
    param_values = [r['param_value'] for r in results]
    first_choice = [r['student_satisfaction']['1st'] for r in results]
    avg_rank = [r['avg_preference_rank'] for r in results]
    fill_rate = [r['school_fill_rate'] for r in results]
    avg_score = [r['avg_score'] for r in results]

    # Plot 1: First choice matches
    axes[0, 0].plot(param_values, first_choice, 'o-')
    axes[0, 0].set_title("Students Matched to 1st Choice")
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel("Count")

    # Plot 2: Average preference rank
    axes[0, 1].plot(param_values, avg_rank, 'o-')
    axes[0, 1].set_title("Average Preference Rank")
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel("Rank (lower is better)")

    # Plot 3: School fill rate
    axes[1, 0].plot(param_values, fill_rate, 'o-')
    axes[1, 0].set_title("School Fill Rate")
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel("Fill Rate")

    # Plot 4: Average score
    axes[1, 1].plot(param_values, avg_score, 'o-')
    axes[1, 1].set_title("Average Score of Matched Students")
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].set_ylabel("Average Score")

    plt.tight_layout()


# ---------- Data Management ----------

def load_yaml_data(filepath: str) -> Tuple[List[Student], Dict[str, School]]:
    """Load student and school data from YAML file."""
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

    students = [Student(s['name'], s['preferences'], s['score'], s['city'])
                for s in data['students']]

    schools = {s['name']: School(s['name'], s['capacity'], s['city'])
               for s in data['schools']}

    return students, schools


def generate_random_data(num_students: int = 20, num_schools: int = 5) -> Tuple[List[Student], Dict[str, School]]:
    """Generate random data for simulation and testing."""
    cities = ["New York", "Boston", "Chicago", "San Francisco", "Austin"]

    # Generate schools
    schools = {}
    for i in range(num_schools):
        name = f"School {i + 1}"
        capacity = random.randint(2, 5)
        city = random.choice(cities)
        schools[name] = School(name, capacity, city)

    school_names = list(schools.keys())

    # Generate students
    students = []
    for i in range(num_students):
        name = f"Student {i + 1}"
        score = random.randint(70, 100)
        city = random.choice(cities)

        # Create preferences (shuffled but weighted by school capacity)
        prefs = random.sample(school_names, min(3, len(school_names)))

        students.append(Student(name, prefs, score, city))

    return students, schools


# ---------- Parameter Exploration ----------

def explore_city_bonus(students: List[Student], schools: Dict[str, School],
                       min_bonus: int = 0, max_bonus: int = 20, steps: int = 5):
    """Run matching with different city bonus values and collect results."""
    results = []

    for bonus in range(min_bonus, max_bonus + 1, max(1, (max_bonus - min_bonus) // steps)):
        # Reset all matches
        for s in students:
            s.match = None
        for sch in schools.values():
            sch.matches = []
            sch.applicants = []

        # Run matching
        gale_shapley(students, schools, bonus)

        # Calculate metrics
        metrics = calculate_satisfaction(students, schools)
        metrics['param_value'] = bonus
        results.append(metrics)

    return results


def explore_capacity(students: List[Student], schools: Dict[str, School],
                     capacity_multiplier_range: Tuple[float, float] = (0.5, 2.0),
                     steps: int = 5):
    """Explore how changing school capacities affects matching."""
    original_capacities = {name: sch.capacity for name, sch in schools.items()}
    results = []

    for mult in [
        capacity_multiplier_range[0] + i * (capacity_multiplier_range[1] - capacity_multiplier_range[0]) / steps
        for i in range(steps + 1)]:
        # Adjust capacities
        for name, sch in schools.items():
            sch.capacity = max(1, math.floor(original_capacities[name] * mult))

        # Reset all matches
        for s in students:
            s.match = None
        for sch in schools.values():
            sch.matches = []
            sch.applicants = []

        # Run matching
        gale_shapley(students, schools, city_bonus=5)

        # Calculate metrics
        metrics = calculate_satisfaction(students, schools)
        metrics['param_value'] = mult
        results.append(metrics)

    # Restore original capacities
    for name, sch in schools.items():
        sch.capacity = original_capacities[name]

    return results


# ---------- GUI Components ----------

class CollegeAdmissionsApp:
    """Main application GUI for college admissions matching."""

    def __init__(self, root):
        self.root = root
        self.root.title("College Admissions Matching - Game Theory Project")

        self.students = []
        self.schools = {}

        self.create_widgets()

    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Data controls
        data_frame = ttk.LabelFrame(main_frame, text="Data Management", padding="10")
        data_frame.grid(row=0, column=0, sticky="ew", pady=5)

        ttk.Button(data_frame, text="Load YAML Data", command=self.load_file).grid(row=0, column=0, padx=5)
        ttk.Button(data_frame, text="Manual Input", command=self.manual_entry).grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Generate Random Data", command=self.generate_random).grid(row=0, column=2, padx=5)

        # Algorithm controls
        algo_frame = ttk.LabelFrame(main_frame, text="Matching Algorithm", padding="10")
        algo_frame.grid(row=1, column=0, sticky="ew", pady=5)

        ttk.Label(algo_frame, text="City Bonus:").grid(row=0, column=0)
        self.city_bonus_var = tk.IntVar(value=5)
        ttk.Spinbox(algo_frame, from_=0, to=20, textvariable=self.city_bonus_var).grid(row=0, column=1)

        ttk.Button(algo_frame, text="Run Gale-Shapley", command=self.run_gale_shapley).grid(row=0, column=2, padx=5)
        ttk.Button(algo_frame, text="Run Random Matching", command=self.run_random_matching).grid(row=0, column=3,
                                                                                                  padx=5)

        # Analysis controls
        analysis_frame = ttk.LabelFrame(main_frame, text="Parameter Analysis", padding="10")
        analysis_frame.grid(row=2, column=0, sticky="ew", pady=5)

        ttk.Button(analysis_frame, text="Explore City Bonus",
                   command=lambda: self.explore_parameter("city_bonus")).grid(row=0, column=0, padx=5)
        ttk.Button(analysis_frame, text="Explore Capacity",
                   command=lambda: self.explore_parameter("capacity")).grid(row=0, column=1, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(main_frame, textvariable=self.status_var, relief="sunken").grid(row=3, column=0, sticky="ew", pady=5)

    def load_file(self):
        """Load data from YAML file."""
        filepath = filedialog.askopenfilename(filetypes=[("YAML files", "*.yml *.yaml")])
        if not filepath:
            return

        try:
            self.students, self.schools = load_yaml_data(filepath)
            self.status_var.set(f"Loaded {len(self.students)} students and {len(self.schools)} schools")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def manual_entry(self):
        """Handle manual data entry."""
        try:
            # Simplified manual entry for demo purposes
            num_students = simpledialog.askinteger("Input", "Number of students:", parent=self.root)
            if not num_students:
                return

            num_schools = simpledialog.askinteger("Input", "Number of schools:", parent=self.root)
            if not num_schools:
                return

            self.students, self.schools = generate_random_data(num_students, num_schools)
            self.status_var.set(f"Created {num_students} students and {num_schools} schools")
        except Exception as e:
            messagebox.showerror("Error", f"Manual input failed: {str(e)}")

    def generate_random(self):
        """Generate random data."""
        try:
            self.students, self.schools = generate_random_data()
            self.status_var.set(f"Generated {len(self.students)} students and {len(self.schools)} schools")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")

    def reset_matches(self):
        """Reset all matching assignments."""
        for student in self.students:
            student.match = None
        for school in self.schools.values():
            school.matches = []
            school.applicants = []

    def run_gale_shapley(self):
        """Run Gale-Shapley algorithm and show results."""
        if not self.students or not self.schools:
            messagebox.showwarning("Warning", "No data loaded")
            return

        self.reset_matches()
        city_bonus = self.city_bonus_var.get()

        gale_shapley(self.students, self.schools, city_bonus)

        # Show report
        report = generate_report(self.students, self.schools)
        messagebox.showinfo("Matching Results - Gale-Shapley", report)

        # Show visualizations
        satisfaction = calculate_satisfaction(self.students, self.schools)['student_satisfaction']
        plot_satisfaction(satisfaction, "Gale-Shapley Matching Results")
        plot_matching_graph(self.students, self.schools)
        plt.show()

    def run_random_matching(self):
        """Run random matching for comparison."""
        if not self.students or not self.schools:
            messagebox.showwarning("Warning", "No data loaded")
            return

        self.reset_matches()
        random_matching(self.students, self.schools)

        # Show report
        report = generate_report(self.students, self.schools)
        messagebox.showinfo("Matching Results - Random", report)

        # Show visualizations
        satisfaction = calculate_satisfaction(self.students, self.schools)['student_satisfaction']
        plot_satisfaction(satisfaction, "Random Matching Results")
        plot_matching_graph(self.students, self.schools)
        plt.show()

    def explore_parameter(self, param_name: str):
        """Run parameter exploration and show results."""
        if not self.students or not self.schools:
            messagebox.showwarning("Warning", "No data loaded")
            return

        if param_name == "city_bonus":
            results = explore_city_bonus(self.students.copy(), self.schools.copy())
            plot_parameter_analysis(results, "City Priority Bonus")
        elif param_name == "capacity":
            results = explore_capacity(self.students.copy(), self.schools.copy())
            plot_parameter_analysis(results, "Capacity Multiplier")

        plt.show()
        self.status_var.set(f"Completed {param_name} parameter analysis")


# ---------- Main Execution ----------

if __name__ == "__main__":
    root = tk.Tk()
    app = CollegeAdmissionsApp(root)
    root.mainloop()