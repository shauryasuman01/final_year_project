import pandas as pd
import random

columns = [
    "Name", "Roll No", "Course",
    "What did you like most about this course and why?",
    "Which topics were most useful for your understanding or future career?",
    "Which topics were difficult or unclear? Please mention briefly.",
    "How effective was the teaching method used in this course?",
    "Were the lectures and study materials helpful? Explain shortly.",
    "How can this course be improved for future students?",
    "Did this course meet your expectations? Why or why not?",
    "How was the pace of teaching (too fast, slow, or balanced)? Explain briefly.",
    "What practical skills or knowledge did you gain from this course?",
    "Any other suggestions or comments?",
    "Ground_Truth_Label"
]

# --- POSITIVE POOL ---
pos_liked = [
    "I loved the practical labs and the way concepts were explained.",
    "The instructor's enthusiasm was contagious.",
    "Everything was perfect, especially the project work.",
    "The clarity of the lectures was the best part.",
    "I liked the real-world applications shown."
]
pos_useful = [
    "The entire syllabus was very relevant.",
    "Especially the advanced modules were great.",
    "The coding sessions were extremely useful.",
    "All topics were well-chosen for our career.",
    "Data structures and algorithms were taught well."
]
pos_difficult = [
    "None, everything was clear.",
    "I found it all easy to understand.",
    "Nothing was too difficult, the teacher helped a lot.",
    "No topics were unclear.",
    "Everything was explained perfectly."
]
pos_teaching = [
    "The teaching method was excellent and engaging.",
    "Very effective, I learned a lot.",
    "Top-notch teaching style.",
    "Interactive and very helpful.",
    "Perfect balance of theory and practice."
]
pos_materials = [
    "Yes, the materials were very comprehensive.",
    "The slides and notes were perfect.",
    "Yes, very helpful resources.",
    "The study descriptions were great.",
    "Everything provided was useful."
]
pos_improvements = [
    "No improvements needed, it is great.",
    "Just keep doing what you are doing.",
    "Nothing to change.",
    "It is already perfect.",
    "Maybe just more of the same!"
]
pos_expectations = [
    "Yes, it exceeded my expectations.",
    "Absolutely, I am very happy.",
    "Yes, learned more than I thought.",
    "Completely met my expectations.",
    "Yes, it was a fantastic course."
]
pos_pace = [
    "The pace was perfect.",
    "Very balanced.",
    "Just right.",
    "Good pace, easy to follow.",
    "Excellent pacing."
]
pos_skills = [
    "Gained solid coding skills.",
    "Learned a lot about system design.",
    "Great practical knowledge gained.",
    "Confident in applying these skills now.",
    "Mastered the core concepts."
]
pos_suggestions = [
    "Great job!",
    "Thank you for this course.",
    "Keep it up.",
    "Best course ever.",
    "Loved it."
]

# --- NEGATIVE POOL ---
neg_liked = [
    "Nothing really, the course was disappointing.",
    "I didn't like anything.",
    "Not much, it was very boring.",
    "The only good thing was when it ended.",
    "Nothing stood out as positive."
]
neg_useful = [
    "None of the topics seemed useful.",
    "I don't think I learned anything relevant.",
    "Most topics were outdated.",
    "Nothing was practical.",
    "Hard to say, it was all confusing."
]
neg_difficult = [
    "Almost everything was unclear.",
    "The entire syllabus was too hard to follow.",
    "I didn't understand the core concepts.",
    "The math part was impossible to grasp.",
    "Nothing was explained well."
]
neg_teaching = [
    "The teaching method was terrible.",
    "Very ineffective, just reading slides.",
    "Boring and unengaging.",
    "The teacher didn't explain anything properly.",
    "Not effective at all."
]
neg_materials = [
    "No, the materials were useless.",
    "Notes were confusing and unstructured.",
    "Not helpful at all.",
    "We didn't get proper study materials.",
    "The book suggested was bad."
]
neg_improvements = [
    "Change the teacher.",
    "Update the entire syllabus.",
    "Make it more interactive, it's too boring.",
    "Teach actual practical skills.",
    "Everything needs improvement."
]
neg_expectations = [
    "No, it was a waste of time.",
    "Not at all, very disappointed.",
    "No, I expected to learn more.",
    "It failed to meet any expectation.",
    "No, it was worse than I thought."
]
neg_pace = [
    "Way too fast, couldn't keep up.",
    "Too slow and boring.",
    "Very uneven and confusing.",
    "Rushed through important topics.",
    "Too slow, wasted time on basics."
]
neg_skills = [
    "I didn't gain any new skills.",
    "Nothing practical.",
    "I am more confused now.",
    "Zero practical knowledge.",
    "None."
]
neg_suggestions = [
    "Please hire better instructors.",
    "Don't offer this course if you can't teach it.",
    "Very bad experience.",
    "Please improve significantly.",
    "I will not recommend this."
]

first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan", "Diya", "Saanvi", "Ananya", "Aadhya", "Pari"]
last_names = ["Sharma", "Verma", "Gupta", "Malhotra", "Bhatia", "Mehta", "Joshi", "Nair", "Patel", "Reddy", "Singh", "Kumar"]
courses = ["Quantum Computing", "Machine Learning", "Data Structures", "Operating Systems"]

data = []

# Generate 100 Positive
for i in range(100):
    row = [
        f"{random.choice(first_names)} {random.choice(last_names)}",
        f"R2024{100+i}",
        random.choice(courses),
        random.choice(pos_liked),
        random.choice(pos_useful),
        random.choice(pos_difficult),
        random.choice(pos_teaching),
        random.choice(pos_materials),
        random.choice(pos_improvements),
        random.choice(pos_expectations),
        random.choice(pos_pace),
        random.choice(pos_skills),
        random.choice(pos_suggestions),
        1 # Ground Truth Positive
    ]
    data.append(row)

# Generate 100 Negative
for i in range(100):
    row = [
        f"{random.choice(first_names)} {random.choice(last_names)}",
        f"R2024{200+i}",
        random.choice(courses),
        random.choice(neg_liked),
        random.choice(neg_useful),
        random.choice(neg_difficult),
        random.choice(neg_teaching),
        random.choice(neg_materials),
        random.choice(neg_improvements),
        random.choice(neg_expectations),
        random.choice(neg_pace),
        random.choice(neg_skills),
        random.choice(neg_suggestions),
        0 # Ground Truth Negative
    ]
    data.append(row)

df = pd.DataFrame(data, columns=columns)
df = df.sample(frac=1).reset_index(drop=True)
output_path = 'd:/final_year_project/final_year_project/student_feedback.csv'
df.to_csv(output_path, index=False)
print(f"Generated 200 polarized samples with Ground Truth at {output_path}")
