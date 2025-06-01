from app import db, app
import bcrypt
from datetime import datetime
from flask_login import UserMixin
import os


# Class to define User model for storing account information
class User(db.Model, UserMixin):
    __tablename__ = 'user'
    # table columns
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    wallet_address = db.Column(db.String(100), nullable=False)
    register_date = db.Column(db.DateTime, nullable=False)
    current_login = db.Column(db.DateTime, nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)
    # Foreign key linking to Role
    role_id = db.Column(db.Integer, db.ForeignKey('role.role_id'), nullable=False)
    # Relationship (Many Users → One Role)
    role = db.relationship('Role', backref='users')

    def __init__(self, username, email, password, wallet_address, role):
        self.username = username
        self.email = email
        # slow hash algo to store and salt hash of passwords in db
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        self.wallet_address = wallet_address
        self.register_date = datetime.now()
        self.current_login = None
        self.last_login = None
        self.role = role


# Class to define roles for role-based access control
class Role(db.Model):
    __tablename__ = 'role'
    # table columns
    role_id = db.Column(db.Integer, primary_key=True)
    role_name = db.Column(db.String(10), nullable=False, unique=True)

    # initialise role
    def __init__(self, role_name):
        self.role_name = role_name


# function to create a role and add it to the Role DB table
def create_role(name):
    role = Role(role_name=name)
    db.session.add(role)
    db.session.commit()


# change user role
def change_role(user, new_role):
    ch_user = User.query.filter_by(username=user).first()
    role = Role.query.filter_by(role_name=new_role).first()
    ch_user.role = role
    db.session.commit()


# initialise db
def init_db():
    with app.app_context():
        db.drop_all()
        db.create_all()

        # create default roles
        create_role('admin')
        create_role('user')
        create_role('client')

        # default admin user will eventually be created with role admin
        admin_role = Role.query.filter_by(role_name='admin').first()
        admin = User(username='admin',
                     email='admin@email.com',
                     password='Admin123!',
                     wallet_address=os.getenv('CONTRACT_OWNER_ADDR'),
                     role=admin_role)
        # for role in admin.roles:
        #     print(role.role_name)
        db.session.add(admin)
        db.session.commit()
