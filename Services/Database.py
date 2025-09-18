from sqlmodel import SQLModel, Field, create_engine, Session

DATABASE_URL = "mysql+mysqlconnector://root:Sujith%406212@localhost:3306/knot"
engine = create_engine(DATABASE_URL)

class Task(SQLModel, table=True):
    id: int = Field(primary_key=True)
    task_date: str
    title: str
    user_email: str

# Create table if not exists
def init_db():
    SQLModel.metadata.create_all(engine)

# Get session
def get_session():
    return Session(engine)
