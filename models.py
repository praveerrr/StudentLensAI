"""ORM model for the students table."""
from sqlalchemy import Column, String, Integer, Float
from database import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=False, index=True)
    semester = Column(Integer, nullable=False, index=True)

    study_hours = Column(Float, nullable=False)
    attendance = Column(Float, nullable=False)
    previous_score = Column(Float, nullable=False)
    assignment_score = Column(Float, nullable=False)
    internal_marks = Column(Float, nullable=False)
    sleep_hours = Column(Float, nullable=False)
    participation = Column(Float, nullable=False)

    final_score = Column(Float, nullable=False)
    risk = Column(String, nullable=False, index=True)

    def as_feature_vector(self):
        return [
            self.study_hours, self.attendance, self.previous_score,
            self.assignment_score, self.internal_marks, self.sleep_hours,
            self.participation,
        ]

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "department": self.department,
            "semester": self.semester, "study_hours": self.study_hours,
            "attendance": self.attendance, "previous_score": self.previous_score,
            "assignment_score": self.assignment_score, "internal_marks": self.internal_marks,
            "sleep_hours": self.sleep_hours, "participation": self.participation,
            "final_score": self.final_score, "risk": self.risk,
        }
