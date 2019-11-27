from src import db


class User(db.Model):
    """
    User table
    """
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(255))

    def __init__(self, username="", email="", password=""):
        self.username = username
        self.email = email
        self.password = password

    @staticmethod
    def is_active():
        return True

    @staticmethod
    def is_authenticated():
        return True

    def get_id(self):
        return self.id

    def __repr__(self):
        return '<User %r>' % self.username
