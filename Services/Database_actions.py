from Services.Database import Task, get_session

def create_task(task_date, title, user_email):
    with get_session() as session:
        task = Task(task_date=task_date, title=title, user_email=user_email)
        session.add(task)
        session.commit()
        session.refresh(task)
        return task

def update_task(task_id, title=None, task_date=None, user_email=None):
    with get_session() as session:
        task = session.get(Task, task_id)
        if not task:
            return None
        if title: task.title = title
        if task_date: task.task_date = task_date
        if user_email: task.user_email = user_email
        session.add(task)
        session.commit()
        return task

def delete_task(task_id):
    with get_session() as session:
        task = session.get(Task, task_id)
        if not task:
            return None
        session.delete(task)
        session.commit()
        return task

def fetch_tasks(user_email=None):
    with get_session() as session:
        if user_email:
            return session.query(Task).filter(Task.user_email == user_email).all()
        return session.query(Task).all()
